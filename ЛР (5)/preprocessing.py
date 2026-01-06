"""
Модуль для предобработки данных: обработка пропущенных значений и преобразование категориальных данных.

Этот модуль предоставляет класс DataPreprocessor для универсальной обработки данных,
который можно использовать для различных датасетов.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


class DataPreprocessor:
    """
    Класс для предобработки данных: обработка пропущенных значений и преобразование категориальных данных.
    
    Поддерживает:
    - Обработку пропущенных значений (медиана для численных, мода/None для категориальных)
    - Ординальное кодирование порядковых признаков
    - One-Hot Encoding номинальных категориальных признаков
    - Сохранение состояния для применения к новым данным (fit/transform pattern)
    """
    
    def __init__(
        self,
        fill_numerical_strategy: str = 'median',
        fill_categorical_strategy: str = 'mode',
        drop_first: bool = True,
        verbose: bool = True
    ):
        """
        Инициализация препроцессора.
        
        Parameters
        ----------
        fill_numerical_strategy : str, default='median'
            Стратегия заполнения пропусков в численных признаках.
            Возможные значения: 'median', 'mean', 'mode'
        fill_categorical_strategy : str, default='mode'
            Стратегия заполнения пропусков в категориальных признаках.
            Возможные значения: 'mode', 'constant'
        drop_first : bool, default=True
            Удалять ли первый столбец при One-Hot Encoding для избежания мультиколлинеарности.
        verbose : bool, default=True
            Выводить ли информацию о процессе обработки.
        """
        self.fill_numerical_strategy = fill_numerical_strategy
        self.fill_categorical_strategy = fill_categorical_strategy
        self.drop_first = drop_first
        self.verbose = verbose
        
        # Внутреннее состояние (заполняется при fit)
        self.numerical_cols_ = None
        self.categorical_cols_ = None
        self.numerical_fill_values_ = {}
        self.categorical_fill_values_ = {}
        self.ordinal_encoders_ = {}
        self.onehot_columns_ = []
        self.feature_names_ = None
        self.is_fitted_ = False
        
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        ordinal_features: Optional[Dict[str, List[str]]] = None,
        na_means_none: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None
    ) -> 'DataPreprocessor':
        """
        Обучение препроцессора на обучающих данных.
        
        Parameters
        ----------
        X : pd.DataFrame
            Обучающие данные.
        y : pd.Series, optional
            Целевая переменная (не используется, но может быть передана для совместимости).
        numerical_cols : list of str, optional
            Список численных признаков. Если None, определяется автоматически.
        categorical_cols : list of str, optional
            Список категориальных признаков. Если None, определяется автоматически.
        ordinal_features : dict, optional
            Словарь порядковых признаков с их категориями.
            Формат: {'feature_name': ['cat1', 'cat2', ...]}
        na_means_none : list of str, optional
            Список признаков, где "NA" означает "None" (отсутствие объекта).
        exclude_cols : list of str, optional
            Список столбцов, которые нужно исключить из обработки (например, 'Id', 'SalePrice').
        
        Returns
        -------
        self : DataPreprocessor
            Возвращает self для цепочки вызовов.
        """
        X = X.copy()
        
        # Определяем исключаемые столбцы
        if exclude_cols is None:
            exclude_cols = []
        
        # Определяем численные и категориальные признаки
        if numerical_cols is None:
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        self.numerical_cols_ = numerical_cols
        self.categorical_cols_ = categorical_cols
        
        if self.verbose:
            print("=" * 60)
            print("ОБУЧЕНИЕ ПРЕПРОЦЕССОРА")
            print("=" * 60)
            print(f"\nЧисленные признаки: {len(self.numerical_cols_)}")
            print(f"Категориальные признаки: {len(self.categorical_cols_)}")
        
        # 1. Обработка пропущенных значений - вычисляем значения для заполнения
        self._fit_missing_values(X, na_means_none)
        
        # 2. Обработка ординального кодирования - сохраняем маппинги
        if ordinal_features is not None:
            self._fit_ordinal_encoding(ordinal_features)
        
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"\n✓ Препроцессор обучен")
        
        return self
    
    def transform(
        self,
        X: pd.DataFrame,
        ordinal_features: Optional[Dict[str, List[str]]] = None,
        na_means_none: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Применение преобразований к данным.
        
        Parameters
        ----------
        X : pd.DataFrame
            Данные для преобразования.
        ordinal_features : dict, optional
            Словарь порядковых признаков (должен совпадать с переданным в fit).
        na_means_none : list of str, optional
            Список признаков, где "NA" означает "None" (должен совпадать с переданным в fit).
        
        Returns
        -------
        pd.DataFrame
            Преобразованные данные.
        """
        if not self.is_fitted_:
            raise ValueError("Препроцессор не обучен. Вызовите fit() перед transform().")
        
        if self.verbose:
            print("=" * 60)
            print("ПРИМЕНЕНИЕ ПРЕОБРАЗОВАНИЯ К ДАННЫМ")
            print("=" * 60)
        
        X = X.copy()
        
        # 1. Обработка пропущенных значений
        X = self._transform_missing_values(X, na_means_none)
        
        # 2. Ординальное кодирование
        X = self._transform_ordinal_encoding(X, ordinal_features)
        
        # 3. One-Hot Encoding
        if ordinal_features is not None:
            categorical_for_onehot = [
                col for col in self.categorical_cols_
                if col not in ordinal_features.keys() and col in X.columns
            ]
        else:
            categorical_for_onehot = [
                col for col in self.categorical_cols_ if col in X.columns
            ]
        
        X, onehot_cols = self._transform_onehot_encoding(X, categorical_for_onehot, fit=False)
        
        # Определяем feature_names_ при первом вызове transform
        if self.feature_names_ is None:
            self.feature_names_ = X.columns.tolist()
            self.onehot_columns_ = onehot_cols
        else:
            # Убеждаемся, что порядок и набор столбцов совпадает с обучением
            missing_cols = set(self.feature_names_) - set(X.columns)
            if missing_cols:
                # Добавляем отсутствующие столбцы с нулями
                for col in missing_cols:
                    X[col] = 0
            
            extra_cols = set(X.columns) - set(self.feature_names_)
            if extra_cols:
                # Удаляем лишние столбцы
                X = X.drop(columns=list(extra_cols))
            
            # Упорядочиваем столбцы
            X = X[self.feature_names_]
        
        return X
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        ordinal_features: Optional[Dict[str, List[str]]] = None,
        na_means_none: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
        skip_categorical_encoding: bool = False
    ) -> pd.DataFrame:
        """
        Обучение и применение преобразований за один шаг.
        
        Parameters
        ----------
        X : pd.DataFrame
            Данные для обучения и преобразования.
        y : pd.Series, optional
            Целевая переменная (не используется).
        numerical_cols : list of str, optional
            Список численных признаков.
        categorical_cols : list of str, optional
            Список категориальных признаков.
        ordinal_features : dict, optional
            Словарь порядковых признаков с их категориями.
        na_means_none : list of str, optional
            Список признаков, где "NA" означает "None".
        exclude_cols : list of str, optional
            Список столбцов для исключения.
        skip_categorical_encoding : bool, default=False
            Пропустить преобразование категориальных данных (только обработка пропусков).
        
        Returns
        -------
        pd.DataFrame
            Преобразованные данные.
        """
        if skip_categorical_encoding:
            # Только обработка пропусков
            self.fit(X, y, numerical_cols, categorical_cols,
                    None, na_means_none, exclude_cols)  # ordinal_features=None
            X_transformed = self._transform_missing_values(X.copy(), na_means_none)
            # Обновляем feature_names_ для совместимости
            self.feature_names_ = X_transformed.columns.tolist()
            return X_transformed
        else:
            return self.fit(
                X, y, numerical_cols, categorical_cols,
                ordinal_features, na_means_none, exclude_cols
            ).transform(X, ordinal_features, na_means_none)
    
    def _fit_missing_values(
        self,
        X: pd.DataFrame,
        na_means_none: Optional[List[str]] = None
    ):
        """Вычисление значений для заполнения пропусков."""
        if self.verbose:
            print("\nВЫЧИСЛЕНИЕ ЗНАЧЕНИЙ ДЛЯ ЗАПОЛНЕНИЯ ПРОПУСКОВ")
            print("-" * 60)
        
        # Численные признаки
        for col in self.numerical_cols_:
            if col in X.columns and X[col].isnull().sum() > 0:
                if self.fill_numerical_strategy == 'median':
                    fill_value = X[col].median()
                elif self.fill_numerical_strategy == 'mean':
                    fill_value = X[col].mean()
                elif self.fill_numerical_strategy == 'mode':
                    mode_val = X[col].mode()
                    fill_value = mode_val[0] if len(mode_val) > 0 else 0
                else:
                    fill_value = 0
                
                self.numerical_fill_values_[col] = fill_value
                
                if self.verbose:
                    print(f"  {col}: {self.fill_numerical_strategy} = {fill_value:.2f}")
        
        # Категориальные признаки
        for col in self.categorical_cols_:
            if col in X.columns:
                nan_count = X[col].isnull().sum()
                
                if nan_count > 0:
                    if na_means_none is not None and col in na_means_none:
                        # Для признаков, где "NA" означает "None"
                        fill_value = 'None'
                    else:
                        # Используем моду
                        mode_val = X[col].mode()
                        if len(mode_val) > 0:
                            fill_value = mode_val[0]
                        else:
                            fill_value = 'Unknown'
                    
                    self.categorical_fill_values_[col] = fill_value
                    
                    if self.verbose:
                        print(f"  {col}: заполнение = '{fill_value}'")
    
    def _transform_missing_values(
        self,
        X: pd.DataFrame,
        na_means_none: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Применение обработки пропущенных значений."""
        if self.verbose:
            print("\nОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
            print("-" * 60)
        
        # Обработка численных признаков
        for col in self.numerical_cols_:
            if col in X.columns and X[col].isnull().sum() > 0:
                if col in self.numerical_fill_values_:
                    fill_value = self.numerical_fill_values_[col]
                else:
                    # Если признак не был в обучающих данных, используем медиану из текущих данных
                    fill_value = X[col].median()
                
                missing_count = X[col].isnull().sum()
                X[col].fillna(fill_value, inplace=True)
                
                if self.verbose:
                    print(f"  {col}: заполнено {missing_count} пропусков ({self.fill_numerical_strategy} = {fill_value:.2f})")
        
        # Обработка категориальных признаков
        for col in self.categorical_cols_:
            if col in X.columns:
                nan_count = X[col].isnull().sum()
                
                # Заменяем строковые "NA" на "None" для признаков, где это уместно
                if na_means_none is not None and col in na_means_none:
                    na_string_count = (X[col] == 'NA').sum() if X[col].dtype == 'object' else 0
                    if na_string_count > 0:
                        X[col] = X[col].replace('NA', 'None')
                        if self.verbose:
                            print(f"  {col}: заменено {na_string_count} значений 'NA' на 'None'")
                
                if nan_count > 0:
                    if na_means_none is not None and col in na_means_none:
                        fill_value = 'None'
                    elif col in self.categorical_fill_values_:
                        fill_value = self.categorical_fill_values_[col]
                    else:
                        # Если признак не был в обучающих данных, используем моду из текущих данных
                        mode_val = X[col].mode()
                        fill_value = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    
                    X[col].fillna(fill_value, inplace=True)
                    
                    if self.verbose:
                        print(f"  {col}: заполнено {nan_count} пропусков значением '{fill_value}'")
        
        return X
    
    def _fit_ordinal_encoding(
        self,
        ordinal_features: Dict[str, List[str]]
    ):
        """Сохранение маппингов для ординального кодирования."""
        if self.verbose:
            print("\nСОХРАНЕНИЕ МАППИНГОВ ДЛЯ ОРДИНАЛЬНОГО КОДИРОВАНИЯ")
            print("-" * 60)
        
        for col, categories in ordinal_features.items():
            if col in self.categorical_cols_:
                category_map = {cat: idx for idx, cat in enumerate(categories)}
                self.ordinal_encoders_[col] = category_map
                
                if self.verbose:
                    print(f"  {col}: сохранен маппинг для {len(categories)} категорий")
    
    def _transform_ordinal_encoding(
        self,
        X: pd.DataFrame,
        ordinal_features: Optional[Dict[str, List[str]]]
    ) -> pd.DataFrame:
        """Применение ординального кодирования."""
        if ordinal_features is None or len(ordinal_features) == 0:
            return X
        
        if self.verbose:
            print("\nПРИМЕНЕНИЕ ОРДИНАЛЬНОГО КОДИРОВАНИЯ")
            print("-" * 60)
        
        for col, categories in ordinal_features.items():
            if col in X.columns and col in self.ordinal_encoders_:
                category_map = self.ordinal_encoders_[col]
                
                # Применяем кодирование
                X[col] = X[col].map(category_map)
                
                # Заполняем неизвестные значения медианой
                if X[col].isnull().sum() > 0:
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    X[col].fillna(median_val, inplace=True)
                
                X[col] = X[col].astype(int)
                
                if self.verbose:
                    print(f"  {col}: закодировано {len(categories)} категорий")
        
        return X
    
    def _transform_onehot_encoding(
        self,
        X: pd.DataFrame,
        categorical_cols: List[str],
        fit: bool = False
    ) -> tuple:
        """Применение One-Hot Encoding."""
        if len(categorical_cols) == 0:
            return X, []
        
        if self.verbose:
            print("\nПРИМЕНЕНИЕ ONE-HOT КОДИРОВАНИЯ")
            print("-" * 60)
        
        # Фильтруем только те столбцы, которые есть в данных
        categorical_cols = [col for col in categorical_cols if col in X.columns]
        
        if len(categorical_cols) == 0:
            return X, []
        
        # Применяем One-Hot Encoding
        encoded_cols = pd.get_dummies(
            X[categorical_cols],
            prefix=categorical_cols,
            drop_first=self.drop_first,
            dummy_na=False,
            dtype=int
        )
        
        # Удаляем исходные категориальные столбцы
        X = X.drop(columns=categorical_cols)
        
        # Добавляем закодированные столбцы
        X = pd.concat([X, encoded_cols], axis=1)
        
        if self.verbose:
            print(f"  Создано {len(encoded_cols.columns)} новых бинарных признаков")
            if len(encoded_cols.columns) > 0:
                print(f"  Примеры: {list(encoded_cols.columns[:5])}...")
        
        return X, encoded_cols.columns.tolist()

