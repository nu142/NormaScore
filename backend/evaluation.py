import pandas as pd
from backend.model_call import extract_text, llm_extract_schema
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
    num_fds = len(ref_schema.get('fds', []))
    fd_total = num_fds * 0.5
    
    num_final = len(ref_schema.get('final_tables', []))
    final_total = num_final * 0.5
    
    rubric = {
        "Functional Dependencies": {"max": fd_total, "rule": "0.5 marks per FD"},
        "1NF": {"max": 2.0, "rule": "Composite(1) + Multivalued(1)"},
        "2NF": {"max": 4.0, "rule": "Partial(1) + Lossless(2) + Key(1)"},
        "3NF": {"max": 4.0, "rule": "Transitive(1) + Lossless(2) + Key(1)"},
        "Final Relations": {"max": final_total, "rule": "0.5 marks per Relation"},
        "TOTAL": {"max": fd_total + 2 + 4 + 4 + final_total, "rule": "Sum"}
    }
    
    df = pd.DataFrame.from_dict(rubric, orient='index')
    df.columns = ['Max Marks', 'Scoring Rule']
    return df

class NormalizationEvaluator:
    def __init__(self, ref_schema):
        self.ref = ref_schema
        self.rubric = build_rubric_sheet(ref_schema)
        self.ref_fds_parsed = parse_fds(ref_schema.get('fds', []))
        self.max_score = self.rubric.loc['TOTAL']['Max Marks'] if 'TOTAL' in self.rubric.index else 16.0
        
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
            
        fd_score = matched_fds * 0.5
        total_score += fd_score
        report.append({
            "Step": "Functional Dependencies", 
            "Score": fd_score, 
            "Max": self.rubric.loc['Functional Dependencies']['Max Marks'] if 'Functional Dependencies' in self.rubric.index else 0,
            "Feedback": f"Matched {matched_fds} FDs. Missing: {missing_fds}" if missing_fds else "All FDs identified."
        })

        # 2. 1NF
        s1nf = student_sub.get('1nf', [])
        score_1nf = 0
        feedback_1nf = []
        ref_1nf_attrs = set(self.ref['1nf'][0]['attributes']) if self.ref.get('1nf') and len(self.ref['1nf']) > 0 else set()
        stu_1nf_attrs = set(s1nf[0]['attributes']) if s1nf and len(s1nf) > 0 else set()
        
        if stu_1nf_attrs == ref_1nf_attrs and len(ref_1nf_attrs) > 0:
            score_1nf = 2.0
            feedback_1nf.append("Attributes match reference 1NF.")
        elif len(ref_1nf_attrs) > 0:
            missing = ref_1nf_attrs - stu_1nf_attrs
            extra = stu_1nf_attrs - ref_1nf_attrs
            score_1nf = max(0, 2.0 - (0.5 * len(missing)))
            feedback_1nf.append(f"Attribute mismatch. Missing: {missing}, Extra: {extra}")
        else:
            feedback_1nf.append("No reference 1NF provided.")

        total_score += score_1nf
        report.append({"Step": "1NF", "Score": score_1nf, "Max": 2.0, "Feedback": " ".join(feedback_1nf)})

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
        s2, f2 = compare_stage('2nf', 4.0)
        total_score += s2
        report.append({"Step": "2NF", "Score": s2, "Max": 4.0, "Feedback": f2})

        # 3NF
        s3, f3 = compare_stage('3nf', 4.0)
        total_score += s3
        report.append({"Step": "3NF", "Score": s3, "Max": 4.0, "Feedback": f3})

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
                    
        final_score = final_matches * 0.5
        total_score += final_score
        report.append({
            "Step": "Final Relations", 
            "Score": final_score, 
            "Max": self.rubric.loc['Final Relations']['Max Marks'] if 'Final Relations' in self.rubric.index else 0,
            "Feedback": f"Identified {final_matches}/{len(ref_final)} final tables."
        })
        
        return total_score, report

def evaluate_all(student_files, reference_schema, hf_token=None):
    evaluator = NormalizationEvaluator(reference_schema)
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
                
            percent_score = (score / evaluator.max_score) * 100 if evaluator.max_score > 0 else 0
            results.append({
                "Student Name": name,
                "Score": percent_score,
                "Feedback": feedback_str,
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
