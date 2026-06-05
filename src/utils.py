import logging
import json

logger = logging.getLogger(__name__)

def load_data(path):
    logger.info('Loading data from file...')
    # Synthetic data for demonstration purposes
    data = [{'text': 'Sample text 1'}, {'text': 'Sample text 2'}, {'text': 'Sample text 3'}]
    logger.info('Data loaded successfully')
    return data

def save_results(path, results):
    logger.info('Saving results to file...')
    with open(path, 'w') as f:
        json.dump(results, f)
    logger.info('Results saved successfully')