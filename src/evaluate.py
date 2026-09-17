EVAL_SET = [
    {
        "question": "What is policy reuse in reinforcement learning?",
        "expected_source": "06aamas-policy-reuse.pdf",
    },
    {
        "question": "How does the teacher-student framework work in reinforcement learning?",
        "expected_source": "Teacher-Student Framework A RL Approach.pdf",
    },
    {
        "question": "What is the COMA algorithm and how does it use a counterfactual baseline in multi-agent policy gradients?",
        "expected_source": "Counterfactual Multi-Agent Policy Gradients.pdf",
    },
    {
        "question": "How does DQN achieve human-level control using deep reinforcement learning?",
        "expected_source": "Human-level control through deep reinforcement-DQN.pdf",
    },
    {
        "question": "What conditions must a reward transformation satisfy to preserve policy invariance?",
        "expected_source": "Policy invariance under reward transformation-LAPTOP-R7S6EJL2.pdf",
    },
    {
        "question": "How is social influence used as an intrinsic motivation signal in multi-agent reinforcement learning?",
        "expected_source": "Social influence as intrinsic motivation for MARL (1).pdf",
    },
    {
        "question": "What is hindsight credit assignment in reinforcement learning?",
        "expected_source": "hindsight-credit-assignment-Paper.pdf",
    },
    {
        "question": "How does Align-RUDDER use reward redistribution to learn from few demonstrations?",
        "expected_source": "Align-RUDDER Learning From Few Demonstrations by Reward redistribution.pdf",
    },
]

import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from graph_basic import app as basic_app
from graph_agentic import app as agentic_app

llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key = os.environ["BASERT_KEY"], model=os.environ["BASERT_MODEL"])

def hit_rate(retrieved_docs, expected_source):
    for docs in retrieved_docs:
        if expected_source == docs["metadata"]["file"]:
            return 1.0
    return 0.0
            
def reciprocal_rank(retrieved_docs, expected_source):
    retrieved_docs_array = np.array([docs["metadata"]["file"] for docs in retrieved_docs])
    if expected_source not in retrieved_docs_array:
        return 0
    else:
        return 1/(np.where(retrieved_docs_array == expected_source)[0][0] + 1)

def is_grounded(contexts, answer):
    context_joined = "\n\n".join([context["text"] for context in contexts])
    prompt = f"You are an expert text analyser who looks at a given context {context_joined} and evaluate whether the provided {answer} faithfully grounded"\
            "to the context above. Response with yes if you think adequately grounded no if not"
    reponse = llm.invoke(prompt)
    return reponse.content.strip().lower().startswith("yes") 

def is_relevant_answer(question, answer):
    prompt = f"You are an expert text analyser who looks at a given question {question} and evaluate whether the provided {answer} is a valid answer for the given question."\
         "Response with yes if you think it is a valid, relevant answer to the question, no if not"
            
    reponse = llm.invoke(prompt)
    return reponse.content.strip().lower().startswith("yes") 

def run_eval(app):
    evaluation_result = []
    for i, eval_item in enumerate(EVAL_SET):
        config = {"configurable": {"thread_id": f"eval-{i}"}}
        result = app.invoke({"question": eval_item["question"]}, config=config)
        hit_rate_score = hit_rate(result["retrieved_docs"], eval_item["expected_source"])
        reciprocal_rank_score = reciprocal_rank(result["retrieved_docs"], eval_item["expected_source"])
        is_grounded_score = is_grounded(result["retrieved_docs"], result["answer"])
        is_relevant_answer_score = is_relevant_answer(eval_item["question"], result["answer"])
        evaluation_result.append([hit_rate_score, reciprocal_rank_score, is_grounded_score, is_relevant_answer_score])
        if not is_grounded_score:
            print(f"\n--- UNGROUNDED: {eval_item['question']}")
            print("ANSWER:", result["answer"])
            print("---\n")
    evaluation_result_array = np.array(evaluation_result)
    mean_evaluation_result_array = np.mean(evaluation_result_array, axis=0)
    return mean_evaluation_result_array

if __name__ == "__main__":
    metric_names = ["Hit Rate", "MRR", "Groundedness", "Relevance"]
    print("[INFO] Running evaluation on basic rag")
    basic_scores = run_eval(basic_app)
    print("[INFO] Running evaluation on agentic rag")
    agentic_scores = run_eval(agentic_app)

    print(f"{'Metric':<15}{'Basic':<10}{'Agentic':<10}")
    for name, b, a in zip(metric_names, basic_scores, agentic_scores):
        print(f"{name:<15}{b:<10.3f}{a:<10.3f}")
    

        
        

