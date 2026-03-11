
dataset = {
    "age":        [20, 25, 30, None, 40, 35, None, 28, 50, 60],
    "salary":     [1000, 2000, None, 4000, 5000, None, 7000, 8000, 9000, 10000],
    "score":      [10, 20, 30, 40, None, 60, 70, None, 90, 100],
    "city":       ["SP", "RJ", "MG", "SP", None, "BA", "BA", "MG", "RJ", "SP"],
    "department": ["IT", "HR", "IT", None, "Finance", "IT", "HR", "Finance", "IT", None]
}


colunas = list(dataset.values())

print(colunas)