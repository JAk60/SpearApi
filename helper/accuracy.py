import csv
import os

def accuracy_save_to_csv(output_text, labels, path):
    path = path + "accuracy.csv"
    # Prepare CSV data
    csv_data = {
        'Task': labels,
        'epoch': None,
        'best_epoch': None,
        'score_used': None,
        'best_gm_val_score': None,
        'best_fm_val_score': None,
        'best_gm_test_score': None,
        'best_fm_test_score': None,
        'best_gm_test_precision': None,
        'best_fm_test_precision': None,
        'best_gm_test_recall': None,
        'best_fm_test_recall': None
    }

    # Extract relevant data from output_text
    lines = output_text.strip().split('\n')
    for line in lines:
        items = line.split('\t')
        for item in items:
            key_value = item.split(':')
            key = key_value[0].strip()
            value = key_value[1].strip()
            if key == 'early stopping at epoch':
                csv_data['epoch'] = value
            elif key == 'score used':
                csv_data['score_used'] = value
            elif key in csv_data.keys():
                csv_data[key] = value

    # Check if CSV file already exists
    file_exists = os.path.isfile(path)

    # Write to CSV file
    with open(path, 'a', newline='') as csvfile:
        fieldnames = list(csv_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is newly created
        if not file_exists:
            writer.writeheader()

        # Write row with labels as the first column
        writer.writerow(csv_data)

    print(f'CSV data has been saved to "{path}".')


def accuracy_round_save_to_csv(output_text, labels, path):
    path = os.path.join(path, "accuracy_round.csv")  # Adjusted to ensure proper path formatting
    
    # Prepare CSV data
    csv_data = {
        'Task': labels,
        'epoch': None,
        'best_epoch': None,
        'gm_val_score': None,
        'fm_val_score': None,
        'gm_test_score': None,
        'fm_test_score': None,
        'gm_test_precision': None,
        'fm_test_precision': None,
        'gm_test_recall': None,
        'fm_test_recall': None
    }

    # Split the output_text into lines and process each line
    lines = output_text.strip().split('\n')
    for line in lines:
        # Split each line by tab (\t)
        items = line.split('\t')
        for item in items:
            # Split each item by colon (:)
            key_value = item.split(':')
            if key_value[0].strip() != "best_epoch" and key_value[0].strip() != "score used":
                key = key_value[0].strip().replace('best_', '')  # Remove 'best_' prefix if present
            elif key_value[0].strip() == "best_epoch" :
                key = key_value[0].strip()
            else:
                continue  # Skip 'score used' line

            value = key_value[1].strip()
            
            # Update csv_data based on key
            if key == 'early stopping at epoch':
                csv_data['epoch'] = value
            elif key == 'best_epoch':
                csv_data['best_epoch'] = value
            elif key in csv_data.keys():
                csv_data[key] = value

    # Convert numerical values to percentages (rounded to 2 decimals)
    for key in csv_data.keys():
        if key not in ['Task', 'epoch', 'best_epoch']:
            if csv_data[key] is not None:
                csv_data[key] = f"{float(csv_data[key]) * 100:.2f}"

    # Check if CSV file already exists
    file_exists = os.path.isfile(path)

    # Write to CSV file
    with open(path, 'a', newline='') as csvfile:
        fieldnames = list(csv_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is newly created
        if not file_exists:
            writer.writeheader()

        # Write row with labels as the first column
        writer.writerow(csv_data)

    print(f'CSV data has been saved to "{path}".')


def accuracy_per_save_to_csv(output_text, labels, path):
    path = os.path.join(path, "accuracy_per.csv")  # Adjusted to ensure proper path formatting
    
    # Prepare CSV data
    csv_data = {
        'Task': labels,
        'gm_val_score': None,
        'fm_val_score': None,
        'gm_test_score': None,
        'fm_test_score': None,
        'gm_test_precision': None,
        'fm_test_precision': None,
        'gm_test_recall': None,
        'fm_test_recall': None
    }

    # Split the output_text into lines and process each line
    lines = output_text.strip().split('\n')
    for line in lines:
        # Split each line by tab (\t)
        items = line.split('\t')
        for item in items:
            # Split each item by colon (:)
            key_value = item.split(':')
            key = key_value[0].strip()
            value = key_value[1].strip()
            
            # Update csv_data based on key, excluding 'epoch', 'best_epoch', and 'score_used'
            if key not in ['early stopping at epoch', 'best_epoch', 'score used']:
                key = key.replace('best_', '')  # Remove 'best_' prefix if present
                if key in csv_data.keys():
                    csv_data[key] = value

    # Convert numerical values to percentages (rounded to 2 decimals)
    for key in csv_data.keys():
        if key not in ['Task']:
            if csv_data[key] is not None:
                csv_data[key] = f"{float(csv_data[key]) * 100:.2f}%"

    # Check if CSV file already exists
    file_exists = os.path.isfile(path)

    # Write to CSV file
    with open(path, 'a', newline='') as csvfile:
        fieldnames = list(csv_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is newly created
        if not file_exists:
            writer.writeheader()

        # Write row with labels as the first column
        writer.writerow(csv_data)

    print(f'CSV data has been saved to "{path}".')


import csv

def cage_accuracy_save_to_csv(output_text, labels, path):
    # Initialize variables to store accuracy score and f1 scores
    raw_path = path +"accuracy.csv"
    percentage_path = path + "accuracy_per.csv"
    accuracy_score = None
    f1_scores = {'macro': None, 'micro': None, 'weighted': None}

    # Split output_text into lines and process each line
    lines = output_text.strip().split('\n')
    for line in lines:
        # Split each line by '\t' and extract key-value pairs
        parts = line.split('\t')
        for part in parts:
            key_value = part.split(': ')
            if len(key_value) == 2:
                key = key_value[0].strip()
                value = key_value[1].strip()

                if key == 'final_test_accuracy_score':
                    accuracy_score = float(value)
                elif key == 'final_test_f1_score':
                    if 'macro' in line:
                        f1_scores['macro'] = float(value)
                    elif 'micro' in line:
                        f1_scores['micro'] = float(value)
                    elif 'weighted' in line:
                        f1_scores['weighted'] = float(value)

    # Prepare row data
    raw_row_data = [labels, accuracy_score, f1_scores['macro'], f1_scores['micro'], f1_scores['weighted']]
    percentage_row_data = [
        labels,
        f"{accuracy_score * 100:.2f}%", 
        f"{f1_scores['macro'] * 100:.2f}%", 
        f"{f1_scores['micro'] * 100:.2f}%", 
        f"{f1_scores['weighted'] * 100:.2f}%"
    ]

    # Check if CSV files exist
    raw_csv_exists = False
    percentage_csv_exists = False
    try:
        with open(raw_path, 'r') as f:
            raw_csv_exists = True
    except FileNotFoundError:
        raw_csv_exists = False
    
    try:
        with open(percentage_path, 'r') as f:
            percentage_csv_exists = True
    except FileNotFoundError:
        percentage_csv_exists = False

    # Write to raw scores CSV file, append if it exists
    with open(raw_path, 'a', newline='') as raw_csv_file:
        raw_csv_writer = csv.writer(raw_csv_file)
        if not raw_csv_exists:
            raw_csv_writer.writerow(['Task', 'Accuracy', 'F1 Score (macro)', 'F1 Score (micro)', 'F1 Score (weighted)'])
        raw_csv_writer.writerow(raw_row_data)

    # Write to percentage scores CSV file, append if it exists
    with open(percentage_path, 'a', newline='') as percentage_csv_file:
        percentage_csv_writer = csv.writer(percentage_csv_file)
        if not percentage_csv_exists:
            percentage_csv_writer.writerow(['Task', 'Accuracy (%)', 'F1 Score (macro %)', 'F1 Score (micro %)', 'F1 Score (weighted %)'])
        percentage_csv_writer.writerow(percentage_row_data)

    print(f"Accuracy store: '{raw_path}' and '{percentage_path}' successfully.")





if __name__ == '__main__':

    # Example usage:
    output_text = """
    early stopping at epoch: 19\tbest_epoch: 16
    score used: accuracy_score
    best_gm_val_score:0.9444444444444444\tbest_fm_val_score:0.6111111111111112
    best_gm_test_score:0.8888888888888888\tbest_fm_test_score:0.6666666666666666
    best_gm_test_precision:0.9166666666666666\tbest_fm_test_precision:0.5555555555555556
    best_gm_test_recall:0.8888888888888888\tbest_fm_test_recall:0.6666666666666666
    """
    labels = 'Category'

    accuracy_save_to_csv(output_text, labels, './')
    accuracy_per_save_to_csv(output_text, labels, './')
    accuracy_round_save_to_csv(output_text, labels, './')

    output_text = """final_test_accuracy_score: 0.9305555555555556
    test_average_metric: macro\tfinal_test_f1_score: 0.9191919191919191
    test_average_metric: micro\tfinal_test_f1_score: 0.9305555555555556
    test_average_metric: weighted\tfinal_test_f1_score: 0.9292929292929293
    """

    cage_accuracy_save_to_csv(output_text, labels, "./")
