import pandas as pd
from backend.model_call import extract_text, llm_extract_schema, generate_nlp_feedback
import json

def parse_fds(fd_list):
    parsed = []
    for fd in fd_list:
        if "->" not in fd: continue
        lhs, rhs = fd.split("->")
        parsed.append({
            "lhs": frozenset(a.strip() for a in lhs.split(",")),
            "rhs": frozenset(a.strip() for a in rhs.split(","))
        })
    return parsed

def build_rubric_sheet(ref_schema):
    rubric = {
        "Functional Dependencies": {"max": 1.0, "rule": " marks per FD"},
        "1NF (Composite)": {"max": 1.0, "rule": "Handling of composite attributes"},
        "1NF (Multivalued)": {"max": 1.0, "rule": "Handling of multivalued attributes"},
        "2NF (Partial)": {"max": 1.0, "rule": "Identify and remove partial dependencies"},
        "2NF (Lossless)": {"max": 2.0, "rule": "Ensure lossless decomposition"},
        "2NF (Key)": {"max": 1.0, "rule": "Correct primary keys assigned"},
        "3NF (Transitive)": {"max": 1.0, "rule": "Identify and remove transitive dependencies"},
        "3NF (Lossless)": {"max": 2.0, "rule": "Ensure lossless decomposition"},
        "3NF (Key)": {"max": 1.0, "rule": "Correct primary keys assigned"},
        "Final Relations": {"max": 1.0, "rule": " marks per Relation"}
    }
    
    df = pd.DataFrame.from_dict(rubric, orient='index')
    df.columns = ['Score', 'Scoring Rule']
    return df

class NormalizationEvaluator:
    def __init__(self, ref_schema, custom_rubric=None):
        self.ref = ref_schema
        self.ref_fds_parsed = parse_fds(ref_schema.get('fds', []))
        
        if custom_rubric is not None:
            self.rubric = custom_rubric
        else:
            self.rubric = build_rubric_sheet(ref_schema)
            
        # Calculate max_score correctly based on per-item scores
        fd_per_item = self.rubric.loc['Functional Dependencies']['Score'] if 'Functional Dependencies' in self.rubric.index else 0
        final_per_item = self.rubric.loc['Final Relations']['Score'] if 'Final Relations' in self.rubric.index else 0
        
        fd_total_max = fd_per_item * len(self.ref_fds_parsed)
        final_total_max = final_per_item * len(self.ref.get('final_tables', []))
        
        other_scores = self.rubric.drop(['Functional Dependencies', 'Final Relations'], errors='ignore')['Score'].sum()
        self.max_score = fd_total_max + final_total_max + other_scores
    def evaluate(self, student_sub):
        report = []
        total_score = 0
        
        # 1. FDs
        stu_fds = parse_fds(student_sub.get('fds', []))
        matched_fds = 0
        missing_fds = []
        for rfd in self.ref_fds_parsed:
            found = False
            for sfd in stu_fds:
                if sfd['lhs'] == rfd['lhs'] and sfd['rhs'] == rfd['rhs']:
                    found = True
                    break
            if found: matched_fds += 1
            else: missing_fds.append(f"{set(rfd['lhs'])}->{set(rfd['rhs'])}")
            
        max_fd_per = self.rubric.loc['Functional Dependencies']['Score'] if 'Functional Dependencies' in self.rubric.index else 0
        fd_score = matched_fds * max_fd_per
        total_score += fd_score
        report.append({
            "Step": "Functional Dependencies", 
            "Score": fd_score, 
            "Max": max_fd_per * len(self.ref_fds_parsed),
            "Feedback": f"Matched {matched_fds} FDs. Missing: {missing_fds}" if missing_fds else "All FDs identified."
        })

        # 2. 1NF
        s1nf = student_sub.get('1nf', [])
        score_1nf = 0
        feedback_1nf = []
        ref_1nf_attrs = set(self.ref['1nf'][0]['attributes']) if self.ref.get('1nf') and len(self.ref['1nf']) > 0 else set()
        stu_1nf_attrs = set(s1nf[0]['attributes']) if s1nf and len(s1nf) > 0 else set()
        
        keys_1nf = ['1NF (Composite)', '1NF (Multivalued)']
        valid_1nf_keys = self.rubric.index.intersection(keys_1nf)
        max_1nf = self.rubric.loc[valid_1nf_keys]['Score'].sum() if not valid_1nf_keys.empty else (self.rubric.loc['1NF']['Score'] if '1NF' in self.rubric.index else 2.0)
        if stu_1nf_attrs == ref_1nf_attrs and len(ref_1nf_attrs) > 0:
            score_1nf = max_1nf
            feedback_1nf.append("Attributes match reference 1NF.")
        elif len(ref_1nf_attrs) > 0:
            missing = ref_1nf_attrs - stu_1nf_attrs
            extra = stu_1nf_attrs - ref_1nf_attrs
            penalty_per_mistake = max_1nf / 4.0 if max_1nf > 0 else 0.5
            score_1nf = max(0, max_1nf - (penalty_per_mistake * (len(missing) + len(extra))))
            feedback_1nf.append(f"Attribute mismatch. Missing: {missing}, Extra: {extra}")
        else:
            feedback_1nf.append("No reference 1NF provided.")

        total_score += score_1nf
        report.append({"Step": "1NF", "Score": score_1nf, "Max": max_1nf, "Feedback": " ".join(feedback_1nf)})

        def compare_stage(stage_name, max_score):
            ref_tables = self.ref.get(stage_name, [])
            stu_tables = student_sub.get(stage_name, [])
            
            matches = 0
            for rt in ref_tables:
                rt_attrs = set(rt.get('attributes', []))
                rt_pk = set(rt.get('pk', []))
                for st in stu_tables:
                    if set(st.get('attributes', [])) == rt_attrs:
                        if set(st.get('pk', [])) == rt_pk:
                            matches += 1
                        else:
                            matches += 0.5
                        break
            
            if len(ref_tables) == 0: return max_score, "No tables in this stage."
            
            ratio = matches / len(ref_tables)
            stage_score = ratio * max_score
            
            feedback = f"Matched {matches}/{len(ref_tables)} tables correctly."
            return stage_score, feedback

        # 2NF
        keys_2nf = ['2NF (Partial)', '2NF (Lossless)', '2NF (Key)']
        valid_2nf_keys = self.rubric.index.intersection(keys_2nf)
        max_2nf = self.rubric.loc[valid_2nf_keys]['Score'].sum() if not valid_2nf_keys.empty else (self.rubric.loc['2NF']['Score'] if '2NF' in self.rubric.index else 4.0)
        s2, f2 = compare_stage('2nf', max_2nf)
        total_score += s2
        report.append({"Step": "2NF", "Score": s2, "Max": max_2nf, "Feedback": f2})

        # 3NF
        keys_3nf = ['3NF (Transitive)', '3NF (Lossless)', '3NF (Key)']
        valid_3nf_keys = self.rubric.index.intersection(keys_3nf)
        max_3nf = self.rubric.loc[valid_3nf_keys]['Score'].sum() if not valid_3nf_keys.empty else (self.rubric.loc['3NF']['Score'] if '3NF' in self.rubric.index else 4.0)
        s3, f3 = compare_stage('3nf', max_3nf)
        total_score += s3
        report.append({"Step": "3NF", "Score": s3, "Max": max_3nf, "Feedback": f3})

        # Final
        ref_final = self.ref.get('final_tables', [])
        stu_final = student_sub.get('final_tables', [])
        
        final_matches = 0
        for rt in ref_final:
            rt_attrs = set(rt.get('attributes', []))
            for st in stu_final:
                if set(st.get('attributes', [])) == rt_attrs:
                    final_matches += 1
                    break
                    
        max_final_per = self.rubric.loc['Final Relations']['Score'] if 'Final Relations' in self.rubric.index else 0
        final_score = final_matches * max_final_per
        total_score += final_score
        report.append({
            "Step": "Final Relations", 
            "Score": final_score, 
            "Max": max_final_per * len(ref_final),
            "Feedback": f"Identified {final_matches}/{len(ref_final)} final tables."
        })
        
        return total_score, report

def evaluate_all(student_files, reference_schema, hf_token=None, custom_rubric=None):
    evaluator = NormalizationEvaluator(reference_schema, custom_rubric=custom_rubric)
    results = []
    
    for file in student_files:
        name = file.name
        text = extract_text(file, name)
        
        # Try JSON parsing
        try:
            student_json = json.loads(text)
        except json.JSONDecodeError:
            # Fallback to LLM extraction
            student_json = llm_extract_schema(text, hf_token=hf_token, is_student=True)
            
        if student_json:
            score, report = evaluator.evaluate(student_json)
            # Combine feedback
            feedback_str = " | ".join([f"{r['Step']}: {r['Feedback']}" for r in report if r['Feedback'].strip() and 'All FDs' not in r['Feedback'] and 'match reference' not in r['Feedback']])
            if not feedback_str:
                feedback_str = "Perfect execution. All criteria met beautifully."
                
            # Generate NLP feedback
            nlp_feedback = generate_nlp_feedback(feedback_str, name, score, evaluator.max_score, hf_token)
                
            percent_score = (score / evaluator.max_score) * 100 if evaluator.max_score > 0 else 0
            results.append({
                "Student Name": name,
                "Score": percent_score,
                "Feedback": nlp_feedback,
                "RawScore": score,
                "MaxScore": evaluator.max_score
            })
        else:
            results.append({
                "Student Name": name,
                "Score": 0,
                "Feedback": "Failed to extract database logic from the submission.",
                "RawScore": 0,
                "MaxScore": evaluator.max_score
            })
            
    return pd.DataFrame(results)
