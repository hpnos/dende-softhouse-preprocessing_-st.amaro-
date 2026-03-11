import unittest
from unittest.mock import patch
import copy

from dende_preprocessing import Preprocessing  # ajuste conforme necessário


class TestPreprocessingWith10x5Dataset(unittest.TestCase):

    def setUp(self):
        self.dataset = {
            "age":        [20, 25, 30, None, 40, 35, None, 28, 50, 60],
            "salary":     [1000, 2000, None, 4000, 5000, None, 7000, 8000, 9000, 10000],
            "score":      [10, 20, 30, 40, None, 60, 70, None, 90, 100],
            "city":       ["SP", "RJ", "MG", "SP", None, "BA", "BA", "MG", "RJ", "SP"],
            "department": ["IT", "HR", "IT", None, "Finance", "IT", "HR", "Finance", "IT", None]
        }

    # ==========================================
    # TESTE DE VALIDAÇÃO DO SHAPE
    # ==========================================

    def test_invalid_dataset_shape(self):
        invalid_dataset = {
            "a": [1, 2, 3],
            "b": [1, 2]
        }

        with self.assertRaises(ValueError):
            Preprocessing(invalid_dataset)
        
        # CORREÇÃO: Adicionado caso de borda para detectar inconsistência em listas vazias vs preenchidas
        with self.assertRaises(ValueError):
            Preprocessing({"a": [], "b": [1]})

    # ==========================================
    # TESTE ISNA
    # ==========================================

    def test_isna(self):
        prep = Preprocessing(copy.deepcopy(self.dataset))
        result = prep.isna()

        self.assertTrue(len(result["age"]) > 0)
        self.assertTrue(len(result["age"]) < 10)
        
        # CORREÇÃO: Verificação exata da quantidade de linhas nulas no dataset fornecido (exatamente 7 linhas contêm None)
        self.assertEqual(len(result["age"]), 7)

    # ==========================================
    # TESTE FILLNA
    # ==========================================

    def test_fillna(self):
        prep = Preprocessing(copy.deepcopy(self.dataset))
        result = prep.fillna(value=0)

        for col in result:
            self.assertNotIn(None, result[col])
            
        # CORREÇÃO: Validação para garantir que o None foi de fato substituído pelo valor passado (índice 3 da idade era None)
        self.assertEqual(result["age"][3], 0)

    # ==========================================
    # TESTE DROPNA
    # ==========================================

    def test_dropna(self):
        prep = Preprocessing(copy.deepcopy(self.dataset))
        result = prep.dropna()

        # Todas as linhas retornadas não podem conter None
        for col in result:
            self.assertNotIn(None, result[col])
            
        # CORREÇÃO: Verificação do número de linhas restantes após o dropna (apenas 3 linhas não possuem nenhum None em todo o dataset)
        self.assertEqual(len(result["age"]), 3)

    # ==========================================
    # TESTE MINMAX SCALER
    # ==========================================

    def test_minmax_scaler(self):
        prep = Preprocessing(copy.deepcopy(self.dataset))
        prep.fillna(value=0)

        result = prep.scale(columns={"age", "salary", "score"}, method="minMax")

        for col in ["age", "salary", "score"]:
            self.assertEqual(min(result[col]), 0)
            self.assertEqual(max(result[col]), 1)
            
            # CORREÇÃO: Caso de borda para garantir que todos os valores estão contidos estritamente no intervalo [0, 1]
            self.assertTrue(all(0 <= val <= 1 for val in result[col]))

    # ==========================================
    # TESTE STANDARD SCALER
    # ==========================================

    def test_standard_scaler(self):
        prep = Preprocessing(copy.deepcopy(self.dataset))
        prep.fillna(value=0)

        result = prep.scale(columns={"age"}, method="standard")

        mean = sum(result["age"]) / len(result["age"])
        self.assertAlmostEqual(mean, 0, places=6)
        
        # CORREÇÃO: No Standard Scaler (Z-Score), além da média ser 0, o desvio padrão deve ser obrigatoriamente 1
        variance = sum((x - mean) ** 2 for x in result["age"]) / len(result["age"])
        self.assertAlmostEqual(variance ** 0.5, 1, places=6)

    # ==========================================
    # TESTE LABEL ENCODER
    # ==========================================

    def test_label_encoding(self):
        prep = Preprocessing(copy.deepcopy(self.dataset))
        prep.fillna(value="Unknown")

        result = prep.encode(columns={"city"}, method="label")

        for value in result["city"]:
            self.assertIsInstance(value, int)
            
        # CORREÇÃO: Validação de que os dados foram codificados em um intervalo numérico válido e não em valores soltos ou negativos
        self.assertTrue(all(v >= 0 for v in result["city"]))

    # ==========================================
    # TESTE ONEHOT ENCODER
    # ==========================================

    def test_onehot_encoding(self):
        prep = Preprocessing(copy.deepcopy(self.dataset))
        prep.fillna(value="Unknown")

        result = prep.encode(columns={"department"}, method="oneHot")

        self.assertNotIn("department", result)

        onehot_columns = [col for col in result.keys() if col.startswith("department_")]
        self.assertTrue(len(onehot_columns) > 0)
        
        # CORREÇÃO: A codificação One-Hot deve gerar estritamente valores binários (apenas 0 ou 1) nas novas colunas
        for col in onehot_columns:
            for val in result[col]:
                self.assertIn(val, [0, 1])

    # ==========================================
    # TESTE MÉTODO INVÁLIDO
    # ==========================================

    def test_invalid_scaler_method(self):
        prep = Preprocessing(copy.deepcopy(self.dataset))

        with self.assertRaises(ValueError):
            prep.scale(method="invalid")

    def test_invalid_encoder_method(self):
        prep = Preprocessing(copy.deepcopy(self.dataset))

        with self.assertRaises(ValueError):
            prep.encode(columns={"city"}, method="invalid")

    # ==========================================
    # TESTE DE INTEGRIDADE ESTRUTURAL
    # ==========================================

    def test_dataset_integrity_after_operations(self):
        prep = Preprocessing(copy.deepcopy(self.dataset))
        prep.fillna(value=0)
        prep.scale(columns={"age", "salary"}, method="minMax")
        prep.encode(columns={"city"}, method="label")

        dataset = prep.dataset
        row_count = len(next(iter(dataset.values())))

        for col in dataset:
            self.assertEqual(len(dataset[col]), row_count)


# ==========================================
# MOCK EXEMPLO (caso queira isolar Statistics)
# ==========================================

class TestWithMockedStatistics(unittest.TestCase):

    @patch("dende_preprocessing.Statistics")
    def test_preprocessing_with_mocked_statistics(self, mock_stats):
        dataset = {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "c": [1]*10,
            "d": [2]*10,
            "e": [3]*10
        }

        prep = Preprocessing(dataset)

        self.assertIsNotNone(prep.statistics)
        mock_stats.assert_called_once()


if __name__ == "__main__":
    unittest.main()