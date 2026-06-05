import logging

logger = logging.getLogger(__name__)

class GEvalModel:
    def __init__(self):
        pass

    def score(self, data):
        logger.info('Running G-EVAL scoring...')
        # Synthetic scoring for demonstration purposes
        scores = [0.8, 0.7, 0.9]
        logger.info('G-EVAL scoring completed successfully')
        return scores

class FaithfulnessModel:
    def __init__(self):
        pass

    def score(self, data):
        logger.info('Running faithfulness scoring...')
        # Synthetic scoring for demonstration purposes
        scores = [0.9, 0.8, 0.7]
        logger.info('Faithfulness scoring completed successfully')
        return scores