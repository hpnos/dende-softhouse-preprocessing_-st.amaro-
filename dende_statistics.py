class Statistics:
    def __init__(self, dataset):
        if not isinstance(dataset, dict):
            raise TypeError("O dataset deve ser um dicionário.")
        
        self.dataset = dataset 
        self.ordinal_scales = {}
        colunas = list(self.dataset.keys())

        if not colunas:
            return
        
        primeira_coluna = colunas[0]
        if not isinstance(self.dataset[primeira_coluna], list):
            raise ValueError(f"A coluna '{primeira_coluna}' deve conter uma lista.")
        
        tamanho_esperado = len(self.dataset[primeira_coluna])

        for rotulo in colunas:
            lista_dados = self.dataset[rotulo]

            if not isinstance(lista_dados, list):
                raise ValueError(f"A entrada '{rotulo}' não é uma lista válida.")

            if len(lista_dados) != tamanho_esperado:
                raise ValueError(f"Inconsistência: A coluna '{rotulo}' tem tamanho diferente.")

            if tamanho_esperado > 0:
                primeiro_elemento = lista_dados[0]
                
                if isinstance(primeiro_elemento, (int, float)):
                    tipo_referencia = (int, float)
                else:
                    tipo_referencia = type(primeiro_elemento)

                for elemento in lista_dados:
                    if elemento is None:
                        continue
                    
                    if isinstance(tipo_referencia, tuple):
                        if not isinstance(elemento, tipo_referencia):
                            raise ValueError(f"Tipo misto na coluna '{rotulo}': esperado numérico.")
                    else:
                        if type(elemento) is not tipo_referencia:
                            raise ValueError(f"Tipo misto na coluna '{rotulo}': esperado {tipo_referencia}.")
        
        

    def set_ordinal_scale(self, column, scale_list):
        self.ordinal_scales[column] = scale_list

    def mean(self, column):
        if column not in self.dataset:
            return None
        
        dados = self.dataset[column]
        
        quantidade = len(dados)
        if quantidade == 0:
            return 0

        if not isinstance(dados[0], (int, float)):
            return None

        soma_total = 0
        for valor in dados:
            soma_total += valor
        
        if quantidade == 0:
            return 0
        
        mean = soma_total / quantidade
        return mean

    def median(self, column):
        if column not in self.dataset:
            return None
        
        dados = self.dataset[column]
        
        column_size = len(dados)
        if column_size == 0:
            return None

        if not isinstance(dados[0], (int, float)):
            return None 

        sorted_column = sorted(dados)
        middle_position = column_size // 2

        if column_size % 2 != 0:
            return sorted_column[middle_position]

        return (sorted_column[middle_position - 1] + sorted_column[middle_position]) / 2

    def mode(self, column):
        if column not in self.dataset:
            return None
        
        valores = self.dataset[column]

        if len(valores) == 0:
            return []

        freq = {}
        for v in valores:
            if v in freq:
                freq[v] += 1
            else:
                freq[v] = 1

        maior = 0
        for k in freq:
            if freq[k] > maior:
                maior = freq[k]

        modas = []
        for k in freq:
            if freq[k] == maior:
                modas.append(k)

        return modas


    def variance(self, column):
    
        if column not in self.dataset:
            return None
        
        coluna = self.dataset[column]
        tamanho_coluna = len(coluna)
        
        if tamanho_coluna == 0:
            return 0

        if not isinstance(coluna[0], (int, float)):
            raise TypeError("Variância só funciona com numeros")
        
        media = self.mean(column)
        soma_diferencas_quadradas = 0

        for valor in coluna:
            soma_diferencas_quadradas += (valor - media) ** 2

        variancia = soma_diferencas_quadradas / tamanho_coluna
        return variancia

    def stdev(self, column):
        variancia = self.variance(column)
        
        if variancia is None:
            return None
        
        desvio_padrao = round(variancia ** 0.5, 2)
        return desvio_padrao

    def covariance(self, column_a, column_b):
        if column_a not in self.dataset or column_b not in self.dataset:
            return None
        
        coluna_a = self.dataset[column_a]
        coluna_b = self.dataset[column_b]
        
        tamanho_a = len(coluna_a)
        tamanho_b = len(coluna_b)
        
        if tamanho_a != tamanho_b:
            raise ValueError("As colunas devem ter o mesmo tamanho.")
        
        if tamanho_a == 0:
            return 0

        if not isinstance(coluna_a[0], (int, float)) or not isinstance(coluna_b[0], (int, float)):
            raise TypeError("A covariância só funciona com números.")
        
        media_a = self.mean(column_a)
        media_b = self.mean(column_b)

        soma_produtos_desvios = 0

        for i in range(tamanho_a):
            soma_produtos_desvios += (coluna_a[i] - media_a) * (coluna_b[i] - media_b)

        covariancia = soma_produtos_desvios / tamanho_a
        return covariancia


    def itemset(self, column):
        if column not in self.dataset:
            raise ValueError(f"Coluna '{column}' não encontrada.")
        
        return set(self.dataset[column])

    def absolute_frequency(self, column):
        if column not in self.dataset:
            return {}
        
        data_column = self.dataset[column]
        frequency = {}

        for value in data_column:
            if value in frequency:
                frequency[value] += 1
            else:
                frequency[value] = 1
                
        return frequency


    def relative_frequency(self, column):
        relative_frequency = self.absolute_frequency(column)
        total = 0

        for key in relative_frequency:
            total = total + int(relative_frequency[key])
            
        if total == 0:
            return {}

        for key in relative_frequency:
            relative_frequency[key] = int(relative_frequency[key]) / total

        return relative_frequency


    def cumulative_frequency(self, column, frequency_method='absolute'):
        if frequency_method.lower().strip() not in ['absolute', 'relative']:
            raise ValueError("O método deve ser 'absolute' ou 'relative'.")
            
        if frequency_method.lower().strip() == 'absolute':
            base_frequency = self.absolute_frequency(column)
        else:
            base_frequency = self.relative_frequency(column)

        base_frequency_sorted = self.ordinal_scales.get(column, sorted(base_frequency.keys()))

        cumulative_frequency = {}
        item_anterior = 0

        for key in base_frequency_sorted:
            if key in base_frequency:
                valor_atual = base_frequency[key] + item_anterior
                cumulative_frequency[key] = round(valor_atual, 4) if isinstance(valor_atual, float) else valor_atual
                item_anterior = valor_atual
        
        return cumulative_frequency

    def conditional_probability(self, column, value1, value2):
        if column not in self.dataset:
            return None
            
        dados = self.dataset[column]
        tamanho = len(dados)
        
        if tamanho < 2:
            return 0

        total_condicoes = 0
        sucesso_sequencia = 0

        for i in range(tamanho - 1):
            
            if dados[i] == value2:
                total_condicoes += 1
                
                if dados[i + 1] == value1:
                    sucesso_sequencia += 1

        if total_condicoes == 0:
            return 0

        probabilidade = sucesso_sequencia / total_condicoes
        return probabilidade

    def quartiles(self, column):
        if column not in self.dataset:
            return None
        
        valores_coluna = sorted(self.dataset[column])
        tamanho_coluna = len(valores_coluna)

        if tamanho_coluna == 0:
            return {"Q1": 0, "Q2": 0, "Q3": 0}

        if not isinstance(valores_coluna[0], (int, float)):
            raise TypeError("Os quartis só podem ser calculados com dados numéricos.")

        def calcular_percentil(percentil):
            posicao = (tamanho_coluna + 1) * percentil
            
            indice_flutuante = posicao - 1
            
            index_inteiro = int(indice_flutuante)
            fracao = indice_flutuante - index_inteiro

            if index_inteiro < 0:
                return valores_coluna[0]
            if index_inteiro >= tamanho_coluna - 1:
                return valores_coluna[-1]

            valor_base = valores_coluna[index_inteiro]
            proximo_valor = valores_coluna[index_inteiro + 1]
            
            return valor_base + fracao * (proximo_valor - valor_base)

        q1 = calcular_percentil(0.25)
        q2 = calcular_percentil(0.50)
        q3 = calcular_percentil(0.75)

        return {"Q1": q1, "Q2": q2, "Q3": q3}
    
     
    def histogram(self, column, bins): 

        if column not in self.dataset:
            raise ValueError(f"Coluna '{column}' não encontrada no dataset.")

        valores_coluna = self.dataset[column]
        
        if not valores_coluna:
            return {}
        
        if not isinstance(valores_coluna[0], (int, float)):
            return "Valor Não Numérico."
        
        if bins <= 0:
            raise ValueError("O número de bins deve ser maior que zero.")

        menor_valor = min(valores_coluna)
        maior_valor = max(valores_coluna)

        if menor_valor == maior_valor:
            return { f"Se os valores forem iguais: ({menor_valor}, {maior_valor})": len(valores_coluna)}


        tamanho_bin = (maior_valor - menor_valor) / bins

        limites = []
        for i in range(bins + 1):
            ponto = menor_valor + (i * tamanho_bin)
            limites.append(ponto)

        histograma = {}
        for i in range(bins):
            intervalo = (limites[i], limites[i + 1])
            histograma[intervalo] = 0

        for valor in valores_coluna:
            indice = int((valor - menor_valor) / tamanho_bin)
            
            if indice == bins:
                indice -= 1

            intervalo = (limites[indice], limites[indice + 1])
            histograma[intervalo] += 1

        return histograma
    pass