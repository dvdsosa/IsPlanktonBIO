import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def calculate_pipeline_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Extract data from JSON to a flat list
    results = []
    for status, species_dict in data.items():
        if not isinstance(species_dict, dict):
            continue
        for gt_species, items in species_dict.items():
            for item in items:
                # The JSON stores the real species as the main "key"
                # and the final prediction inside the list object
                results.append({
                    'status': status,
                    'ground_truth': gt_species,
                    'predicted': item.get('predicted'),
                    'file': item.get('file')
                })

    # 2. Convert to Pandas DataFrame
    df = pd.DataFrame(results)

    y_true = df['ground_truth']
    y_pred = df['predicted']

    # 3. Calculate Global Metrics (End-to-End)
    end_to_end_acc = accuracy_score(y_true, y_pred)
    
    # average='macro' calculates the metric for each class and then takes the mean
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    print(f"--- GLOBAL METRICS (END-TO-END) ---")
    print(f"Total Images: {len(df)}")
    print(f"Accuracy:  {end_to_end_acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("  -> How it's calculated: Evaluates the final outcome of the entire pipeline across all images.")
    print("     [JSON Keys]: It ignores the top-level key (pipeline status) and strictly compares the ")
    print("     second-level key (Ground Truth species) against the 'predicted' key stored inside ")
    print("     the image object list, representing the final label assigned by the system.\n")

    # 4. Calculate isolated model metrics (MULTICLASS)
    print(f"--- STAGE METRICS (MULTICLASS) ---")
    
    # ---------------------------------------------------------
    # STAGE 1: Prediction reconstruction
    # ---------------------------------------------------------
    y_pred_s1 = []
    for _, row in df.iterrows():
        if row['status'] == 'unmatched-stage1':
            # If it failed Stage-1, the wrong prediction of model 1 is here
            y_pred_s1.append(row['predicted'])
        else:
            # If it passed Stage-1, we assume model 1 correctly predicted the Ground Truth
            y_pred_s1.append(row['ground_truth'])
            
    s1_acc = accuracy_score(y_true, y_pred_s1)
    s1_prec, s1_rec, s1_f1, _ = precision_recall_fscore_support(y_true, y_pred_s1, average='macro', zero_division=0)
    
    print(f"Stage-1 Accuracy:  {s1_acc*100:.2f}%")
    print(f"Stage-1 Precision: {s1_prec*100:.2f}%")
    print(f"Stage-1 Recall:    {s1_rec*100:.2f}%")
    print(f"Stage-1 F1-Score:  {s1_f1*100:.2f}%")
    print("  -> How it's calculated: Evaluates the multi-class prediction of the Stage-1 model.")
    print("     [JSON Keys]: Uses the top-level key to determine correctness. If the top-level key ")
    print("     is 'unmatched-stage1', it extracts the wrong class from the 'predicted' key. For any ")
    print("     other top-level key (e.g., 'matched', 'touches-border', 'unmatched'), it assumes Stage-1 ")
    print("     was correct and uses the second-level key (Ground Truth) as the successful prediction.\n")

    # ---------------------------------------------------------
    # STAGE 2: Evaluated only on images that reached this stage
    # ---------------------------------------------------------
    stage2_df = df[df['status'].isin(['matched', 'unmatched'])]
    if len(stage2_df) > 0:
        y_true_s2 = stage2_df['ground_truth']
        y_pred_s2 = stage2_df['predicted']
        
        s2_acc = accuracy_score(y_true_s2, y_pred_s2)
        s2_prec, s2_rec, s2_f1, _ = precision_recall_fscore_support(y_true_s2, y_pred_s2, average='macro', zero_division=0)
        
        print(f"Stage-2 Accuracy:  {s2_acc*100:.2f}%")
        print(f"Stage-2 Precision: {s2_prec*100:.2f}%")
        print(f"Stage-2 Recall:    {s2_rec*100:.2f}%")
        print(f"Stage-2 F1-Score:  {s2_f1*100:.2f}%")
        print("  -> How it's calculated: Evaluates the multi-class prediction of the Stage-2 model.")
        print("     [JSON Keys]: It strictly filters the JSON to only evaluate images located under the ")
        print("     top-level keys 'matched' (Stage-2 correct) and 'unmatched' (Stage-2 incorrect). ")
        print("     For these subsets, it compares the second-level key (Ground Truth) against the ")
        print("     'predicted' key to calculate the exact classification metrics.\n")

    # 5. Generate detailed report per class
    print("--- DETAILED REPORT PER CLASS (GLOBAL / END-TO-END) ---")
    print(classification_report(y_true, y_pred, zero_division=0))

# Execution
if __name__ == "__main__":
    calculate_pipeline_metrics('output/inference_results.json')