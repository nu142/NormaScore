import json
import pandas as pd
from backend.model_call import extract_text, llm_extract_schema, llm_generate_reference


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_fds(fd_list):
    """
    Convert a list of 'lhs->rhs' strings into dicts with frozenset lhs/rhs.
    Handles both 'a->b' and 'a->b,c' (multiple RHS attributes).
    """
    parsed = []
    for fd in fd_list:
        if "->" not in fd:
            continue
        lhs_raw, rhs_raw = fd.split("->", 1)
        parsed.append({
            "lhs": frozenset(a.strip() for a in lhs_raw.split(",")),
            "rhs": frozenset(a.strip() for a in rhs_raw.split(",")),
        })
    return parsed


def build_rubric_sheet(ref_schema):
    rubric = {
        "Functional Dependencies": {"max": 1.0, "rule": "1 mark per FD"},
        "1NF (Composite)":         {"max": 1.0, "rule": "Handling of composite attributes"},
        "1NF (Multivalued)":       {"max": 1.0, "rule": "Handling of multivalued attributes"},
        "2NF (Partial)":           {"max": 1.0, "rule": "Identify and remove partial dependencies"},
        "2NF (Lossless)":          {"max": 2.0, "rule": "Ensure lossless decomposition"},
        "2NF (Key)":               {"max": 1.0, "rule": "Correct primary keys assigned"},
        "3NF (Transitive)":        {"max": 1.0, "rule": "Identify and remove transitive dependencies"},
        "3NF (Lossless)":          {"max": 2.0, "rule": "Ensure lossless decomposition"},
        "3NF (Key)":               {"max": 1.0, "rule": "Correct primary keys assigned"},
        "Final Relations":         {"max": 1.0, "rule": "1 mark per Relation"},
    }
    df = pd.DataFrame.from_dict(rubric, orient='index')
    df.columns = ['Score', 'Scoring Rule']
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class NormalizationEvaluator:
    def __init__(self, ref_schema, custom_rubric=None):
        self.ref = ref_schema
        self.ref_fds_parsed = parse_fds(ref_schema.get('fds', []))

        self.rubric = custom_rubric if custom_rubric is not None else build_rubric_sheet(ref_schema)

        # Max score: FDs and final tables are per-item; the rest are flat
        fd_per   = self._rubric_score("Functional Dependencies")
        fin_per  = self._rubric_score("Final Relations")

        fd_total  = fd_per  * len(self.ref_fds_parsed)
        fin_total = fin_per * len(self.ref.get('final_tables', []))

        other = self.rubric.drop(
            ['Functional Dependencies', 'Final Relations'], errors='ignore'
        )['Score'].sum()

        self.max_score = fd_total + fin_total + other

    def _rubric_score(self, key: str, default: float = 0.0) -> float:
        if key in self.rubric.index:
            return float(self.rubric.loc[key]['Score'])
        return default

    # ── main evaluate ────────────────────────────────────────────────────────

    def evaluate(self, student_sub: dict):
        report = []
        total  = 0.0

        # 1. Functional Dependencies
        stu_fds    = parse_fds(student_sub.get('fds', []))
        matched    = 0
        missing    = []
        for rfd in self.ref_fds_parsed:
            found = any(
                sfd['lhs'] == rfd['lhs'] and sfd['rhs'] == rfd['rhs']
                for sfd in stu_fds
            )
            if found:
                matched += 1
            else:
                missing.append(f"{set(rfd['lhs'])}->{set(rfd['rhs'])}")

        fd_per   = self._rubric_score("Functional Dependencies")
        fd_score = matched * fd_per
        fd_max   = fd_per * len(self.ref_fds_parsed)
        total   += fd_score
        report.append({
            "Step":     "Functional Dependencies",
            "Score":    fd_score,
            "Max":      fd_max,
            "Feedback": f"Matched {matched} FDs. Missing: {missing}" if missing
                        else "All FDs identified.",
        })

        # 2. 1NF
        s1nf = student_sub.get('1nf', [])
        ref_1nf_attrs = (
            set(self.ref['1nf'][0]['attributes'])
            if self.ref.get('1nf') else set()
        )
        stu_1nf_attrs = set(s1nf[0]['attributes']) if s1nf else set()

        valid_1nf = self.rubric.index.intersection(['1NF (Composite)', '1NF (Multivalued)'])
        max_1nf = (
            float(self.rubric.loc[valid_1nf]['Score'].sum())
            if not valid_1nf.empty
            else (self._rubric_score('1NF', 2.0))
        )

        if not ref_1nf_attrs:
            score_1nf, fb_1nf = 0.0, "No reference 1NF provided."
        elif stu_1nf_attrs == ref_1nf_attrs:
            score_1nf, fb_1nf = max_1nf, "Attributes match reference 1NF."
        else:
            miss = ref_1nf_attrs - stu_1nf_attrs
            extra = stu_1nf_attrs - ref_1nf_attrs
            penalty = max_1nf / 4.0 if max_1nf else 0.5
            score_1nf = max(0.0, max_1nf - penalty * (len(miss) + len(extra)))
            fb_1nf = f"Attribute mismatch. Missing: {miss}, Extra: {extra}"

        total += score_1nf
        report.append({"Step": "1NF", "Score": score_1nf, "Max": max_1nf, "Feedback": fb_1nf})

        # helper for 2NF and 3NF
        def compare_stage(stage_name, max_score):
            ref_tables = self.ref.get(stage_name, [])
            stu_tables = student_sub.get(stage_name, [])
            if not ref_tables:
                return max_score, "No tables in this stage."
            matches = 0.0
            for rt in ref_tables:
                rt_attrs = set(rt.get('attributes', []))
                rt_pk    = set(rt.get('pk', []))
                for st in stu_tables:
                    if set(st.get('attributes', [])) == rt_attrs:
                        matches += 1.0 if set(st.get('pk', [])) == rt_pk else 0.5
                        break
            ratio = matches / len(ref_tables)
            return ratio * max_score, f"Matched {matches}/{len(ref_tables)} tables correctly."

        # 3. 2NF
        valid_2nf = self.rubric.index.intersection(['2NF (Partial)', '2NF (Lossless)', '2NF (Key)'])
        max_2nf = (
            float(self.rubric.loc[valid_2nf]['Score'].sum())
            if not valid_2nf.empty else self._rubric_score('2NF', 4.0)
        )
        s2, f2 = compare_stage('2nf', max_2nf)
        total += s2
        report.append({"Step": "2NF", "Score": s2, "Max": max_2nf, "Feedback": f2})

        # 4. 3NF
        valid_3nf = self.rubric.index.intersection(['3NF (Transitive)', '3NF (Lossless)', '3NF (Key)'])
        max_3nf = (
            float(self.rubric.loc[valid_3nf]['Score'].sum())
            if not valid_3nf.empty else self._rubric_score('3NF', 4.0)
        )
        s3, f3 = compare_stage('3nf', max_3nf)
        total += s3
        report.append({"Step": "3NF", "Score": s3, "Max": max_3nf, "Feedback": f3})

        # 5. Final Relations
        ref_final = self.ref.get('final_tables', [])
        stu_final = student_sub.get('final_tables', [])
        fin_per   = self._rubric_score("Final Relations")
        fin_matches = sum(
            1 for rt in ref_final
            if any(set(st.get('attributes', [])) == set(rt.get('attributes', []))
                   for st in stu_final)
        )
        fin_score = fin_matches * fin_per
        total    += fin_score
        report.append({
            "Step":     "Final Relations",
            "Score":    fin_score,
            "Max":      fin_per * len(ref_final),
            "Feedback": f"Identified {fin_matches}/{len(ref_final)} final tables.",
        })

        return total, report


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE SCHEMA GENERATION  (entry point for the Streamlit app)
# ─────────────────────────────────────────────────────────────────────────────

def generate_reference_schema(teacher_schema: dict, hf_token=None):
    """
    Call the fine-tuned model to produce a reference normalization schema.

    Parameters
    ----------
    teacher_schema : dict
        Keys: relation_name, attributes, multivalued, fds
        (fds are list of ([lhs...], [rhs...]) tuples — same format as the notebook)

    Returns
    -------
    parsed_schema : dict | None
        The model's JSON output, parsed to a dict.
    raw_output : str
        The raw model text (useful for display / debugging).
    """
    parsed, raw = llm_generate_reference(teacher_schema, hf_token=hf_token)
    return parsed, raw


# ─────────────────────────────────────────────────────────────────────────────
# BATCH EVALUATION  (entry point for the Streamlit app)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(student_files, reference_schema: dict, hf_token=None, custom_rubric=None):
    """
    Evaluate a list of student file uploads against a reference schema.

    Parameters
    ----------
    student_files : list of Streamlit UploadedFile
    reference_schema : dict
        The reference schema produced by generate_reference_schema (or manually built).
    custom_rubric : pd.DataFrame | None

    Returns
    -------
    pd.DataFrame with columns:
        Student Name, Score (%), Feedback, RawScore, MaxScore
    """
    evaluator = NormalizationEvaluator(reference_schema, custom_rubric=custom_rubric)
    results   = []

    for file in student_files:
        name = file.name
        text = extract_text(file, name)

        # Try direct JSON parse first (student submitted a JSON file)
        student_json = None
        raw_model_output = None
        try:
            student_json = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Fall back to LLM extraction
            student_json, raw_model_output = llm_extract_schema(text, hf_token=hf_token)

        if student_json:
            score, report = evaluator.evaluate(student_json)
            feedback_parts = [
                f"{r['Step']}: {r['Feedback']}"
                for r in report
                if r['Feedback'].strip()
                   and 'All FDs' not in r['Feedback']
                   and 'match reference' not in r['Feedback']
            ]
            feedback_str = " | ".join(feedback_parts) if feedback_parts \
                else "Perfect execution. All criteria met."

            pct = (score / evaluator.max_score * 100) if evaluator.max_score > 0 else 0
            results.append({
                "Student Name": name,
                "Score":        round(pct, 2),
                "Feedback":     feedback_str,
                "RawScore":     round(score, 3),
                "MaxScore":     round(evaluator.max_score, 3),
                # Raw model output stored for optional display
                "_raw_model_output": raw_model_output or "(JSON parsed directly)",
            })
        else:
            results.append({
                "Student Name": name,
                "Score":        0,
                "Feedback":     "Failed to extract database logic from submission.",
                "RawScore":     0,
                "MaxScore":     round(evaluator.max_score, 3),
                "_raw_model_output": raw_model_output or "(extraction failed)",
            })

    return pd.DataFrame(results)