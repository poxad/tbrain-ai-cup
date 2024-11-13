import os
import subprocess
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def run_data_preprocess():
    """
    Runs the data preprocessing step using `data_preprocess.py`.
    """
    print("Running data preprocessing...")
    subprocess.run(["python", "Preprocess/data_preprocess.py"], check=True)
    print("Data preprocessing completed.")

def run_retrieval():
    """
    Runs the retrieval step using `retrieval.py`.
    """
    print("Running retrieval...")
    subprocess.run(["python", "Model/retrieval.py"], check=True)
    print("Retrieval completed.")

def main(preprocess, retrieve):
    """
    Main function to orchestrate the workflow.
    :param preprocess: Boolean, whether to run the data preprocessing step.
    :param retrieve: Boolean, whether to run the retrieval step.
    """
    if preprocess:
        run_data_preprocess()

    if retrieve:
        run_retrieval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Challenge pipeline.")

    # Arguments to control the workflow
    parser.add_argument(
        "--preprocess", action="store_true", help="Run the data preprocessing step."
    )
    parser.add_argument(
        "--retrieve", action="store_true", help="Run the retrieval step."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Execute the main function
    main(preprocess=args.preprocess, retrieve=args.retrieve)
