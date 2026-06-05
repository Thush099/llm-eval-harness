import logging
from src.models import GEvalModel, FaithfulnessModel
from src.utils import load_data

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self):
        self.g_eval_model = GEvalModel()
        self.faithfulness_model = FaithfulnessModel()

    def run(self, data):
        logger.info('Running G-EVAL scoring...')
        g_eval_scores = self.g_eval_model.score(data)
        logger.info('G-EVAL scoring completed successfully')

        logger.info('Running faithfulness scoring...')
        faithfulness_scores = self.faithfulness_model.score(data)
        logger.info('Faithfulness scoring completed successfully')

        logger.info('Generating leaderboard...')
        leaderboard = self.generate_leaderboard(g_eval_scores, faithfulness_scores)
        logger.info('Leaderboard generated successfully')

        return leaderboard

    def generate_leaderboard(self, g_eval_scores, faithfulness_scores):
        leaderboard = []
        for i in range(len(g_eval_scores)):
            model_name = f'Model {i+1}'
            g_eval_score = g_eval_scores[i]
            faithfulness_score = faithfulness_scores[i]
            leaderboard.append({
                'name': model_name,
                'g-eval-score': g_eval_score,
                'faithfulness-score': faithfulness_score
            })
        return leaderboard