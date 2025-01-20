import argparse

def compute_roc_curve():
    pass

def results():
    predictions = get_data("predictions.csv")
    compute_roc_curve(predictions)



if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Trainer")
   parser.add_argument("-m", "--model_name", default=DECISION_TREE)
   parser.add_argument("-e", "--csv_path", default=DECISION_TREE)

   args = parser.parse_args()
   results(args.model_name, args.csv_path)
