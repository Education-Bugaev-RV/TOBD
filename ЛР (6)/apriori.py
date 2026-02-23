"""
Модуль для поиска ассоциативных правил по алгоритму Apriori.

Используемые понятия:
- Транзакция (transaction) — одна операция/событие, множество элементов (товаров, признаков).
- Набор элементов (itemset) — подмножество элементов; частый набор — встречается не реже порога minsupport.
- Поддержка (support) — доля транзакций, содержащих данный набор (или его абсолютное число).
- Достоверность (confidence) правила A -> B — support(A ∪ B) / support(A).

Входные данные задаются в виде pandas DataFrame (стандартная транзакционная таблица:
каждая строка — одна транзакция). Результаты возвращаются в виде структур данных
для последующего использования в вызывающем коде.
"""

from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

# Именованный кортеж для ассоциативного правила: антецедент, консеквент, поддержка, достоверность.
# Используется для удобного доступа к полям в вызывающем коде.
from typing import NamedTuple


class AssociationRule(NamedTuple):
    """Одно ассоциативное правило: антецедент -> консеквент с метриками support и confidence."""
    antecedent: frozenset
    consequent: frozenset
    support: float
    confidence: float


def dataframe_to_transactions(
    df: pd.DataFrame,
    binary: bool = True,
    prefix_with_column: bool = False
) -> List[frozenset]:
    """
    Преобразует стандартную транзакционную таблицу (pandas DataFrame) в список транзакций.

    Ожидаемый формат таблицы: каждая строка — одна транзакция (одна операция/событие).
    Число строк равно числу транзакций.

    Параметры
    ---------
    df : pandas.DataFrame
        Транзакционная таблица. Одна строка = одна транзакция.
    binary : bool, по умолчанию True
        - True: бинарная матрица. Столбцы интерпретируются как элементы (товары/признаки).
          В транзакцию входят имена столбцов, у которых в данной строке значение
          истинно или ненулевое (типичная матрица «транзакция × товар»).
          Ячейки с 0, False или NaN считаются отсутствием элемента.
        - False: в ячейках хранятся коды элементов. Транзакция — множество непустых
          (не-NaN) значений ячеек строки, приведённых к строковому типу.
    prefix_with_column : bool, по умолчанию False
        Используется только при binary=False. Если True, элемент формируется как
        «имя_столбца = значение», например «col1 = bread». Это позволяет различать
        одинаковые значения из разных столбцов.

    Возвращает
    ----------
    List[frozenset]
        Список транзакций; порядок совпадает с порядком строк в df.
        Каждая транзакция — frozenset строковых идентификаторов элементов
        (frozenset используется как хешируемый тип для ключей в словарях Apriori).
    """
    transactions: List[frozenset] = []

    if binary:
        # Бинарная матрица: столбцы = элементы, в транзакцию входят столбцы с ненулевым/истинным значением.
        # Ячейки 0, False или NaN считаются отсутствием элемента.
        for _, row in df.iterrows():
            items = frozenset(
                str(col) for col in df.columns
                if pd.notna(row[col]) and row[col] not in (0, False)
            )
            transactions.append(items)
    else:
        # В ячейках — коды элементов; транзакция = множество непустых значений строки (строковый тип).
        for _, row in df.iterrows():
            if prefix_with_column:
                # Элемент в виде «столбец = значение», чтобы различать одинаковые значения из разных столбцов.
                items = frozenset(
                    f"{col} = {v}".strip()
                    for col, v in row.items()
                    if v is not None and pd.notna(v) and str(v).strip() != ''
                )
            else:
                items = frozenset(
                    str(v) for v in row.values
                    if v is not None and pd.notna(v) and str(v).strip() != ''
                )
            transactions.append(items)

    return transactions


def _min_support_count(min_support: Union[float, int], n_transactions: int) -> int:
    """
    Преобразует порог поддержки (доля или абсолютное число) в минимальное число транзакций.

    Если 0 < min_support < 1 — трактуется как доля; если min_support >= 1 — как абсолютное число.
    """
    if n_transactions == 0:
        return 0
    if 0 < min_support < 1:
        return max(1, round(min_support * n_transactions))
    return max(0, int(min_support))


def find_frequent_1_itemsets(
    transactions: List[frozenset],
    min_support_count: int
) -> Dict[frozenset, int]:
    """
    Находит все частые 1-элементные наборы (частые одиночные элементы).

    Один проход по транзакциям; подсчёт вхождений каждого элемента через Counter;
    отсечение по min_support_count.

    Параметры
    ---------
    transactions : List[frozenset]
        Список транзакций.
    min_support_count : int
        Минимальное число транзакций, в которых должен встретиться элемент.

    Возвращает
    ----------
    Dict[frozenset, int]
        Словарь {frozenset({item}): support_count} для каждого частого элемента.
    """
    counter: Counter = Counter()
    for t in transactions:
        for item in t:
            counter[item] += 1
    return {
        frozenset({item}): count
        for item, count in counter.items()
        if count >= min_support_count
    }


def generate_candidates(prev_frequents: Dict[frozenset, int], k: int) -> Set[frozenset]:
    """
    Генерация кандидатов размера k из частых наборов размера k-1 (join-шаг Apriori).

    Кандидаты строятся объединением двух частых (k-1)-наборов с одинаковым префиксом
    длины k-2. Затем применяется pruning: остаются только те кандидаты, у которых
    все подмножества размера k-1 являются частыми.

    Параметры
    ---------
    prev_frequents : Dict[frozenset, int]
        Частые наборы размера k-1 (ключи — frozenset, значения — поддержка).
    k : int
        Размер генерируемых кандидатов (k >= 2).

    Возвращает
    ----------
    Set[frozenset]
        Множество кандидатов размера k.
    """
    prev_itemsets = list(prev_frequents.keys())
    if len(prev_itemsets) < 2:
        return set()
    
    # Сортируем элементы в наборах для воспроизводимого «префикса» при join.
    prev_sorted = [tuple(sorted(s)) for s in prev_itemsets]
    prev_set = set(prev_sorted)
    candidates: Set[frozenset] = set()
    n = len(prev_sorted)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = prev_sorted[i], prev_sorted[j]
            # Join: объединяем, если первые k-2 элемента совпадают.
            if a[: k - 2] != b[: k - 2]:
                continue
            union = frozenset(a) | frozenset(b)
            if len(union) != k:
                continue
            # Pruning: все подмножества размера k-1 должны быть частыми.
            all_frequent = True
            for subset in combinations(union, k - 1):
                if tuple(sorted(subset)) not in prev_set:
                    all_frequent = False
                    break
            if all_frequent:
                candidates.add(union)
    return candidates


def count_candidates(
    transactions: List[frozenset],
    candidates: Set[frozenset],
    min_support_count: int
) -> Dict[frozenset, int]:
    """
    Подсчёт поддержки кандидатов (candidate counting): один проход по транзакциям,
    для каждой транзакции проверяем, какие кандидаты в ней содержатся.
    Отсекаем кандидатов с поддержкой ниже min_support_count.

    Параметры
    ---------
    transactions : List[frozenset]
        Список транзакций.
    candidates : Set[frozenset]
        Кандидаты для подсчёта.
    min_support_count : int
        Минимальная поддержка (абсолютное число транзакций).

    Возвращает
    ----------
    Dict[frozenset, int]
        Словарь {candidate: support_count} только для кандидатов, прошедших порог.
    """
    counter: Dict[frozenset, int] = {c: 0 for c in candidates}
    # Один проход по транзакциям: кандидат входит в транзакцию, если кандидат ⊆ транзакция (c <= t).
    for t in transactions:
        for c in candidates:
            if c <= t:
                counter[c] += 1
    return {c: count for c, count in counter.items() if count >= min_support_count}


def apriori(
    transactions: List[frozenset],
    min_support: Union[float, int],
    max_len: Optional[int] = None
) -> Tuple[Dict[frozenset, int], int]:
    """
    Алгоритм Apriori: находит все частые наборы элементов с поддержкой не ниже min_support.

    Поуровнево строит частые 1-наборы, затем кандидатов размера 2, 3, ... и отсекает
    по поддержке, пока есть частые наборы на текущем уровне (или пока не достигнут max_len).

    Параметры
    ---------
    transactions : List[frozenset]
        Список транзакций (каждая транзакция — frozenset строковых идентификаторов).
    min_support : float или int
        Минимальная поддержка: доля (0 < value < 1) или абсолютное число транзакций (value >= 1).
    max_len : int, опционально
        Максимальный размер частого набора; при достижении цикл прекращается.

    Возвращает
    ----------
    Dict[frozenset, int]
        словарь {itemset: support_count} по всем частым наборам.
    """
    n_transactions = len(transactions)
    if n_transactions == 0:
        return {}
    min_sup_count = _min_support_count(min_support, n_transactions)

    # Уровень 1: частые одиночные элементы.
    frequent = find_frequent_1_itemsets(transactions, min_sup_count)
    all_frequent: Dict[frozenset, int] = dict(frequent)
    k = 2
    while True:
        if (max_len is not None) and (k > max_len):
            break
        candidates = generate_candidates(frequent, k)
        if not candidates:
            break
        frequent = count_candidates(transactions, candidates, min_sup_count)
        if not frequent:
            break
        all_frequent.update(frequent)
        k += 1

    return all_frequent


def compute_supports(
    frequent_itemsets_counts: Dict[frozenset, int],
    n_transactions: int
) -> Dict[frozenset, float]:
    """
    Преобразует абсолютные счётчики частых наборов в относительную поддержку (долю транзакций).

    Параметры
    ---------
    frequent_itemsets_counts : Dict[frozenset, int]
        Словарь {itemset: число транзакций, содержащих itemset}.
    n_transactions : int
        Общее число транзакций (знаменатель).

    Возвращает
    ----------
    Dict[frozenset, float]
        Словарь {itemset: support_fraction}, 0 <= support_fraction <= 1,
        отсортированный по убыванию support_fraction.
    """
    if n_transactions == 0:
        support_items = [(k, 0.0) for k in frequent_itemsets_counts]
    else:
        support_items = [
            (itemset, count / n_transactions)
            for itemset, count in frequent_itemsets_counts.items()
        ]
    # сортируем по понижающемуся support_fraction
    sorted_support_items = sorted(support_items, key=lambda x: x[1])
    return dict(sorted_support_items)


def generate_association_rules(
    frequent_itemsets_counts: Dict[frozenset, int],
    n_transactions: int,
    min_confidence: float
) -> List[AssociationRule]:
    """
    Генерация ассоциативных правил из частых наборов по порогу достоверности.\n
    Для каждого частого набора L размера >= 2 перебираются все непустые подмножества A.\n
    Правило A -> B (где B = L \\ A) добавляется, если confidence = support(L)/support(A) >= min_confidence.

    Параметры
    ---------
    frequent_itemsets_counts : Dict[frozenset, int]
        Словарь частых наборов и их абсолютных поддержок.
    n_transactions : int
        Число транзакций (для вычисления поддержки).
    min_confidence : float
        Минимальная достоверность правила (от 0 до 1).

    Возвращает
    ----------
    List[AssociationRule]
        Список правил (antecedent, consequent, support, confidence),
        отсортированный по убыванию confidence, затем по убыванию support.
    """
    if n_transactions == 0:
        return []
    
    rules: List[AssociationRule] = []
    
    for itemset, count_l in frequent_itemsets_counts.items():
        size = len(itemset)
        
        # Обрабатываем только наборы размера >= 2.
        if size < 2:
            continue
        
        support_l = count_l / n_transactions
        # Все непустые собственные подмножества как антецеденты.
        items = list(itemset)
        for r in range(1, size):
            for antecedent_tuple in combinations(items, r):
                
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                
                count_a = frequent_itemsets_counts.get(antecedent)
                if count_a is None or count_a == 0:
                    continue
                
                # Достоверность: P(L)/P(A) = support(L) / support(A).
                confidence = count_l / count_a
                if confidence >= min_confidence:
                    rules.append(AssociationRule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support=support_l,
                        confidence=confidence
                    ))
    # Сортировка по убыванию достоверности, затем по убыванию поддержки.
    rules.sort(key=lambda r: (-r.confidence, -r.support))
    return rules