import argparse
import logging
from src.pipeline import Pipeline
from src.utils import load_data, save_results

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='LLM Evaluation Harness')
    parser.add_argument('--input-data', type=str, required=True, help='Path to input data')
    parser.add_argument('--output-path', type=str, required=True, help='Path to output results')
    args = parser.parse_args()

    logger.info('Loading data...')
    data = load_data(args.input_data)
    logger.info('Data loaded successfully')

    logger.info('Creating pipeline...')
    pipeline = Pipeline()
    logger.info('Pipeline created successfully')

    logger.info('Running pipeline...')
    results = pipeline.run(data)
    logger.info('Pipeline completed successfully')

    logger.info('Saving results...')
    save_results(args.output_path, results)
    logger.info('Results saved successfully')

if __name__ == '__main__':
    main()