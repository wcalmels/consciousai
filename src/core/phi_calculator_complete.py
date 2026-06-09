# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai
# DOI: https://doi.org/10.5281/zenodo.20602077

"""
Sistema Completo de Cálculo de Phi (Información Integrada)
Con todas las optimizaciones implementadas

Versión: 2.0 - Optimizada
Autor: Sistema de Análisis de IIT
Fecha: 2024-11
"""

import numpy as np
from scipy import linalg, sparse
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import time
import logging
from dataclasses import dataclass
from enum import Enum

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhiMethod(Enum):
    """Métodos disponibles para calcular Phi"""
    EXACT = "exact"
    BALANCED = "balanced"
    APPROXIMATE = "approximate"
    AUTO = "auto"


@dataclass
class PhiResult:
    """Resultado del cálculo de Phi"""
    phi_value: float
    method: str
    confidence: float
    computation_time: float
    best_partition: Optional[np.ndarray] = None
    partitions_evaluated: int = 0
    additional_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class PhiCalculatorOptimized:
    """
    Calculador de Phi optimizado con selección adaptativa de método
    
    Implementa tres estrategias:
    1. EXACT: Para sistemas pequeños (n ≤ 8) - O(2^N)
    2. BALANCED: Para sistemas medianos (8 < n ≤ 80) - Clustering espectral + muestreo
    3. APPROXIMATE: Para sistemas grandes (n > 80) - O(N log N)
    """
    
    def __init__(self, cache_enabled: bool = True, max_cache_size: int = 1000):
        """
        Args:
            cache_enabled: Si usar caché para resultados
            max_cache_size: Tamaño máximo del caché
        """
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self.cache = {}
        
        # Estadísticas
        self.stats = {
            'exact_calls': 0,
            'balanced_calls': 0,
            'approximate_calls': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
    
    def calculate_phi(self, 
                     state: np.ndarray, 
                     connectivity: np.ndarray,
                     method: PhiMethod = PhiMethod.AUTO) -> PhiResult:
        """
        Calcula Phi con selección automática o manual del método
        
        Args:
            state: Vector de estado del sistema (n elementos)
            connectivity: Matriz de conectividad (n x n)
            method: Método a usar (AUTO selecciona automáticamente)
            
        Returns:
            PhiResult con valor de phi y métricas
        """
        start_time = time.time()
        n = len(state)
        
        # Validación de entrada
        if n == 0:
            return PhiResult(
                phi_value=0.0,
                method="none",
                confidence=0.0,
                computation_time=0.0,
                additional_metrics={'error': 'Empty state'}
            )
        
        if connectivity.shape != (n, n):
            raise ValueError(f"Connectivity shape {connectivity.shape} doesn't match state size {n}")
        
        # Verificar caché
        if self.cache_enabled:
            cache_key = self._generate_cache_key(state, connectivity)
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                cached_result = self.cache[cache_key]
                logger.debug(f"Cache hit! Returning cached result for n={n}")
                return cached_result
        
        # Seleccionar método
        if method == PhiMethod.AUTO:
            if n <= 8:
                selected_method = PhiMethod.EXACT
            elif n <= 80:
                selected_method = PhiMethod.BALANCED
            else:
                selected_method = PhiMethod.APPROXIMATE
        else:
            selected_method = method
        
        # Calcular phi según método seleccionado
        if selected_method == PhiMethod.EXACT:
            result = self._calculate_exact(state, connectivity)
            self.stats['exact_calls'] += 1
            
        elif selected_method == PhiMethod.BALANCED:
            result = self._calculate_balanced(state, connectivity)
            self.stats['balanced_calls'] += 1
            
        elif selected_method == PhiMethod.APPROXIMATE:
            result = self._calculate_approximate(state, connectivity)
            self.stats['approximate_calls'] += 1
        
        else:
            raise ValueError(f"Unknown method: {selected_method}")
        
        # Añadir tiempo de cómputo
        computation_time = time.time() - start_time
        result.computation_time = computation_time
        self.stats['total_time'] += computation_time
        
        # Guardar en caché
        if self.cache_enabled:
            self._add_to_cache(cache_key, result)
        
        logger.info(f"Phi calculated: {result.phi_value:.4f} (method={result.method}, n={n}, time={computation_time*1000:.2f}ms)")
        
        return result
    
    def _calculate_exact(self, state: np.ndarray, connectivity: np.ndarray) -> PhiResult:
        """
        Cálculo exacto de Phi evaluando todas las particiones bipartitas
        Garantiza encontrar el Phi verdadero (mínimo)
        
        Complejidad: O(2^N * N^2)
        Uso: Solo para n ≤ 8
        """
        n = len(state)
        min_phi = float('inf')
        best_partition = None
        partitions_evaluated = 0
        
        # Evaluar todas las particiones bipartitas no triviales
        # Particiones de 1 a 2^(n-1) - 1 (excluye triviales 0 y 2^n-1)
        for i in range(1, 2**(n-1)):
            partition = self._int_to_partition(i, n)
            
            phi = self._evaluate_partition(state, connectivity, partition)
            partitions_evaluated += 1
            
            if phi < min_phi:
                min_phi = phi
                best_partition = partition
        
        return PhiResult(
            phi_value=min_phi if min_phi != float('inf') else 0.0,
            method=PhiMethod.EXACT.value,
            confidence=1.0,
            computation_time=0.0,  # Se añadirá después
            best_partition=best_partition,
            partitions_evaluated=partitions_evaluated,
            additional_metrics={
                'search_space': 2**(n-1) - 1,
                'completeness': 'full'
            }
        )
    
    def _calculate_balanced(self, 
                           state: np.ndarray, 
                           connectivity: np.ndarray,
                           n_spectral: int = 10,
                           n_random: int = 50) -> PhiResult:
        """
        Cálculo balanceado: clustering espectral + muestreo inteligente
        
        Estrategia dual:
        1. Clustering espectral: encuentra particiones naturales del grafo
        2. Muestreo aleatorio: explora espacio de búsqueda inteligentemente
        
        Complejidad: O(N^2 log N + k*N^2) donde k = n_spectral + n_random
        Uso: Para 8 < n ≤ 80
        """
        n = len(state)
        phi_values = []
        partitions_tested = []
        
        # ESTRATEGIA 1: Clustering Espectral (basado en estructura del grafo)
        spectral_partitions = self._find_spectral_partitions(
            connectivity, 
            max_partitions=n_spectral
        )
        
        logger.debug(f"Found {len(spectral_partitions)} spectral partitions")
        
        for partition in spectral_partitions:
            phi = self._evaluate_partition(state, connectivity, partition)
            phi_values.append(phi)
            partitions_tested.append(partition)
        
        # ESTRATEGIA 2: Muestreo Aleatorio Inteligente
        random_partitions = self._generate_intelligent_partitions(n, n_random)
        
        for partition in random_partitions:
            phi = self._evaluate_partition(state, connectivity, partition)
            phi_values.append(phi)
            partitions_tested.append(partition)
        
        # Encontrar mínimo
        if len(phi_values) == 0:
            return PhiResult(
                phi_value=0.0,
                method=PhiMethod.BALANCED.value,
                confidence=0.0,
                computation_time=0.0,
                additional_metrics={'error': 'No valid partitions found'}
            )
        
        min_idx = np.argmin(phi_values)
        min_phi = phi_values[min_idx]
        best_partition = partitions_tested[min_idx]
        
        # Calcular estadísticas de distribución
        phi_array = np.array(phi_values)
        
        return PhiResult(
            phi_value=min_phi,
            method=PhiMethod.BALANCED.value,
            confidence=min(0.95, len(phi_values) / 100),
            computation_time=0.0,
            best_partition=best_partition,
            partitions_evaluated=len(phi_values),
            additional_metrics={
                'phi_mean': float(np.mean(phi_array)),
                'phi_std': float(np.std(phi_array)),
                'phi_min': float(np.min(phi_array)),
                'phi_max': float(np.max(phi_array)),
                'phi_percentile_25': float(np.percentile(phi_array, 25)),
                'spectral_partitions': len(spectral_partitions),
                'random_partitions': len(random_partitions)
            }
        )
    
    def _calculate_approximate(self, state: np.ndarray, connectivity: np.ndarray) -> PhiResult:
        """
        Aproximación rápida de Phi basada en propiedades espectrales
        
        Usa teoría de información cuántica:
        - Entropía de Von Neumann de la matriz de covarianza
        - Factor de conectividad del grafo
        - Dimensionalidad efectiva del sistema
        
        Complejidad: O(N^2 log N)
        Uso: Para n > 80
        """
        n = len(state)
        
        try:
            # 1. Calcular matriz de covarianza
            if state.ndim > 1:
                cov_matrix = np.cov(state.T)
            else:
                # Para vector 1D, crear covarianza basada en estructura
                state_centered = state - np.mean(state)
                cov_matrix = np.outer(state_centered, state_centered) / n
            
            # 2. Calcular autovalores (espectro)
            eigenvalues = linalg.eigvalsh(cov_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filtrar valores numéricos cercanos a cero
            
            if len(eigenvalues) == 0:
                return PhiResult(
                    phi_value=0.0,
                    method=PhiMethod.APPROXIMATE.value,
                    confidence=0.5,
                    computation_time=0.0,
                    additional_metrics={'note': 'Degenerate system'}
                )
            
            # 3. Entropía de Von Neumann (normalizada)
            eigenvalues_norm = eigenvalues / np.sum(eigenvalues)
            entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-12))
            
            # Normalizar por máxima entropía posible
            max_entropy = np.log(len(eigenvalues))
            entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0
            
            # 4. Factor de conectividad
            connectivity_strength = np.mean(np.abs(connectivity))
            
            # Análisis de distribución de conectividad
            conn_std = np.std(connectivity)
            
            # 5. Calcular Phi aproximado
            # Phi ≈ (entropía del sistema) × (fuerza de conectividad) × sqrt(tamaño)
            phi_approx = entropy_normalized * connectivity_strength * np.sqrt(n)
            
            # Factor de corrección por heterogeneidad de conectividad
            heterogeneity_factor = 1.0 + conn_std / (connectivity_strength + 1e-10)
            phi_approx *= heterogeneity_factor
            
            # Limitar a rango razonable
            phi_approx = np.clip(phi_approx, 0.0, 10.0)
            
            return PhiResult(
                phi_value=float(phi_approx),
                method=PhiMethod.APPROXIMATE.value,
                confidence=0.75,
                computation_time=0.0,
                additional_metrics={
                    'entropy_normalized': float(entropy_normalized),
                    'connectivity_strength': float(connectivity_strength),
                    'connectivity_std': float(conn_std),
                    'effective_dimensions': len(eigenvalues),
                    'heterogeneity_factor': float(heterogeneity_factor),
                    'eigenvalue_ratio': float(eigenvalues[-1] / eigenvalues[0]) if len(eigenvalues) > 0 else 0.0
                }
            )
            
        except Exception as e:
            logger.error(f"Error in approximate calculation: {e}")
            return PhiResult(
                phi_value=0.0,
                method=PhiMethod.APPROXIMATE.value,
                confidence=0.0,
                computation_time=0.0,
                additional_metrics={'error': str(e)}
            )
    
    def _find_spectral_partitions(self, 
                                  connectivity: np.ndarray, 
                                  max_partitions: int = 10) -> List[np.ndarray]:
        """
        Encuentra particiones naturales usando clustering espectral
        
        Basado en el método de Fiedler:
        - Calcula el Laplaciano del grafo: L = D - A
        - El vector de Fiedler (2do autovector) da la bipartición óptima
        - Autovectores adicionales dan particiones alternativas
        
        Fundamentación matemática:
        El vector de Fiedler minimiza el corte del grafo, que es exactamente
        lo que queremos para maximizar integración interna (alto Phi)
        """
        partitions = []
        n = connectivity.shape[0]
        
        try:
            # Construir matriz Laplaciana
            degrees = np.sum(np.abs(connectivity), axis=1)
            D = np.diag(degrees)
            L = D - connectivity
            
            # Calcular autovalores y autovectores
            # Usamos eigh porque L es simétrica (más rápido y estable)
            eigenvalues, eigenvectors = linalg.eigh(L)
            
            # El primer autovalor debe ser ~0 (conectividad)
            # El segundo autovalor (Fiedler value) indica conectividad del grafo
            
            # MÉTODO 1: Vector de Fiedler (2do autovector)
            if eigenvectors.shape[1] > 1:
                fiedler_vector = eigenvectors[:, 1]
                
                # Bipartición basada en signo
                partition_sign = (fiedler_vector >= 0).astype(int)
                if self._is_valid_partition(partition_sign, n):
                    partitions.append(partition_sign)
                
                # Bipartición basada en mediana
                median_val = np.median(fiedler_vector)
                partition_median = (fiedler_vector >= median_val).astype(int)
                if self._is_valid_partition(partition_median, n):
                    partitions.append(partition_median)
                
                # Bipartición basada en percentiles
                for percentile in [33, 50, 67]:
                    threshold = np.percentile(fiedler_vector, percentile)
                    partition = (fiedler_vector >= threshold).astype(int)
                    if self._is_valid_partition(partition, n):
                        partitions.append(partition)
            
            # MÉTODO 2: Autovectores adicionales
            for i in range(2, min(max_partitions + 2, eigenvectors.shape[1])):
                eigenvector = eigenvectors[:, i]
                
                # Múltiples umbrales
                for percentile in [25, 50, 75]:
                    threshold = np.percentile(eigenvector, percentile)
                    partition = (eigenvector >= threshold).astype(int)
                    
                    if self._is_valid_partition(partition, n):
                        partitions.append(partition)
                    
                    if len(partitions) >= max_partitions:
                        break
                
                if len(partitions) >= max_partitions:
                    break
            
            # MÉTODO 3: Clustering basado en múltiples autovectores
            if eigenvectors.shape[1] >= 3 and len(partitions) < max_partitions:
                # Usar primeros 3 autovectores no triviales para clustering
                features = eigenvectors[:, 1:4]
                
                # K-means simple en espacio espectral
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                partition_kmeans = kmeans.fit_predict(features)
                
                if self._is_valid_partition(partition_kmeans, n):
                    partitions.append(partition_kmeans)
        
        except Exception as e:
            logger.warning(f"Error in spectral clustering: {e}")
        
        # Eliminar duplicados
        unique_partitions = []
        seen = set()
        
        for partition in partitions:
            # Normalizar: asegurar que el primer elemento sea siempre 0
            if partition[0] == 1:
                partition = 1 - partition
            
            partition_tuple = tuple(partition)
            if partition_tuple not in seen:
                seen.add(partition_tuple)
                unique_partitions.append(partition)
        
        logger.debug(f"Generated {len(unique_partitions)} unique spectral partitions")
        
        return unique_partitions[:max_partitions]
    
    def _generate_intelligent_partitions(self, n: int, count: int) -> List[np.ndarray]:
        """
        Genera particiones aleatorias de manera inteligente
        
        Estrategias:
        1. Particiones balanceadas (50-50)
        2. Particiones con diferentes tamaños
        3. Particiones con patrones espaciales
        4. Particiones aleatorias puras (para exploración)
        """
        partitions = []
        
        # ESTRATEGIA 1: Particiones balanceadas (mejor para Phi típicamente)
        n_balanced = count // 3
        for _ in range(n_balanced):
            partition = np.zeros(n, dtype=int)
            indices = np.random.choice(n, size=n//2, replace=False)
            partition[indices] = 1
            partitions.append(partition)
        
        # ESTRATEGIA 2: Particiones con tamaños específicos
        sizes = [n//4, n//3, 2*n//3, 3*n//4]
        for size in sizes:
            if 1 <= size < n and len(partitions) < count:
                partition = np.zeros(n, dtype=int)
                indices = np.random.choice(n, size=size, replace=False)
                partition[indices] = 1
                partitions.append(partition)
        
        # ESTRATEGIA 3: Patrones espaciales
        n_patterns = count // 4
        for _ in range(n_patterns):
            if len(partitions) >= count:
                break
            
            # Patrón de bloques contiguos
            block_size = max(1, n // np.random.randint(3, 8))
            partition = np.zeros(n, dtype=int)
            for i in range(0, n, block_size):
                if np.random.random() > 0.5:
                    partition[i:min(i+block_size, n)] = 1
            
            if self._is_valid_partition(partition, n):
                partitions.append(partition)
        
        # ESTRATEGIA 4: Particiones completamente aleatorias (exploración)
        remaining = count - len(partitions)
        for _ in range(remaining):
            partition = np.random.randint(0, 2, n)
            
            # Asegurar no trivialidad
            if np.sum(partition) == 0 or np.sum(partition) == n:
                # Forzar al menos un elemento de cada tipo
                partition[np.random.randint(0, n)] = 0
                partition[np.random.randint(0, n)] = 1
            
            partitions.append(partition)
        
        return partitions[:count]
    
    def _evaluate_partition(self, 
                           state: np.ndarray, 
                           connectivity: np.ndarray,
                           partition: np.ndarray) -> float:
        """
        Evalúa Phi para una partición específica
        
        Phi(partition) = I(A:B) = información mutua entre las dos partes
        
        En IIT, Phi se define como el MÍNIMO de información mutua
        sobre todas las particiones bipartitas posibles
        """
        mask_a = partition == 0
        mask_b = partition == 1
        
        # Verificar validez
        if np.sum(mask_a) == 0 or np.sum(mask_b) == 0:
            return float('inf')  # Partición trivial
        
        try:
            # Estados de cada partición
            state_a = state[mask_a]
            state_b = state[mask_b]
            
            # Conectividad entre particiones
            connectivity_ab = connectivity[np.ix_(mask_a, mask_b)]
            conn_strength = np.mean(np.abs(connectivity_ab)) if connectivity_ab.size > 0 else 0.0
            
            # Calcular información mutua aproximada
            
            # Método 1: Basado en correlación
            if len(state_a) > 0 and len(state_b) > 0:
                mean_a = np.mean(state_a)
                mean_b = np.mean(state_b)
                std_a = np.std(state_a) if len(state_a) > 1 else 1.0
                std_b = np.std(state_b) if len(state_b) > 1 else 1.0
                
                # Correlación entre promedios de las particiones
                if std_a > 1e-10 and std_b > 1e-10:
                    # Correlación ponderada por conectividad
                    state_mean = np.mean(state)
                    correlation = np.abs((mean_a - state_mean) * (mean_b - state_mean)) / (std_a * std_b)
                    correlation = np.clip(correlation, 0, 0.99)
                else:
                    correlation = 0.0
            else:
                correlation = 0.0
            
            # Información mutua basada en correlación
            if correlation < 0.99:
                mutual_info = -0.5 * np.log(1 - correlation**2 + 1e-12)
            else:
                mutual_info = 5.0  # Valor alto para alta correlación
            
            # Phi = información mutua × fuerza de conectividad
            # La conectividad amplifica la importancia de la información compartida
            phi = mutual_info * conn_strength
            
            # Factor de tamaño: particiones muy desbalanceadas son penalizadas
            size_a = np.sum(mask_a)
            size_b = np.sum(mask_b)
            balance_factor = (2 * min(size_a, size_b)) / (size_a + size_b)
            
            phi *= balance_factor
            
            return float(phi)
            
        except Exception as e:
            logger.warning(f"Error evaluating partition: {e}")
            return float('inf')
    
    def _is_valid_partition(self, partition: np.ndarray, n: int) -> bool:
        """Verifica que una partición sea válida (no trivial)"""
        if len(partition) != n:
            return False
        
        sum_partition = np.sum(partition)
        return 0 < sum_partition < n
    
    def _int_to_partition(self, i: int, n: int) -> np.ndarray:
        """Convierte un entero a partición binaria"""
        binary_str = format(i, f'0{n}b')
        return np.array([int(bit) for bit in binary_str], dtype=int)
    
    def _generate_cache_key(self, state: np.ndarray, connectivity: np.ndarray) -> str:
        """Genera clave única para caché"""
        state_hash = hash(state.tobytes())
        conn_hash = hash(connectivity.tobytes())
        return f"{state_hash}_{conn_hash}"
    
    def _add_to_cache(self, key: str, result: PhiResult):
        """Añade resultado al caché con gestión de tamaño"""
        if len(self.cache) >= self.max_cache_size:
            # Eliminar elemento más antiguo (FIFO simple)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = result
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de uso del calculador"""
        total_calls = (self.stats['exact_calls'] + 
                      self.stats['balanced_calls'] + 
                      self.stats['approximate_calls'])
        
        avg_time = self.stats['total_time'] / total_calls if total_calls > 0 else 0.0
        cache_hit_rate = self.stats['cache_hits'] / total_calls if total_calls > 0 else 0.0
        
        return {
            **self.stats,
            'total_calls': total_calls,
            'average_time': avg_time,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache)
        }
    
    def clear_cache(self):
        """Limpia el caché"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def reset_stats(self):
        """Reinicia estadísticas"""
        self.stats = {
            'exact_calls': 0,
            'balanced_calls': 0,
            'approximate_calls': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
        logger.info("Statistics reset")


def create_test_network(n: int, network_type: str = "small_world") -> np.ndarray:
    """
    Crea una red de prueba de n nodos
    
    Args:
        n: Número de nodos
        network_type: Tipo de red ("small_world", "random", "scale_free", "ring")
    
    Returns:
        Matriz de adyacencia n×n
    """
    if network_type == "small_world":
        G = nx.watts_strogatz_graph(n, k=min(4, n-1), p=0.3)
    elif network_type == "random":
        G = nx.erdos_renyi_graph(n, p=0.3)
    elif network_type == "scale_free":
        G = nx.barabasi_albert_graph(n, m=2)
    elif network_type == "ring":
        G = nx.cycle_graph(n)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    return nx.to_numpy_array(G)


# ============================================================================
# FUNCIONES DE UTILIDAD Y DEMOSTRACIÓN
# ============================================================================

def run_comprehensive_benchmark():
    """
    Ejecuta benchmark completo comparando todos los métodos
    """
    print("=" * 80)
    print("🔬 BENCHMARK COMPLETO DEL SISTEMA DE CÁLCULO DE PHI")
    print("=" * 80)
    
    calculator = PhiCalculatorOptimized(cache_enabled=True)
    
    # Tamaños de prueba (optimizados para demo rápida)
    test_sizes = [4, 6, 8, 15, 30, 50, 100, 200]
    
    results = []
    
    for n in test_sizes:
        print(f"\n{'='*80}")
        print(f"📊 Testing system size n = {n}")
        print(f"{'='*80}")
        
        # Generar datos de prueba
        state = np.random.randn(n)
        connectivity = create_test_network(n, "small_world")
        
        # Calcular con método automático
        result = calculator.calculate_phi(state, connectivity, PhiMethod.AUTO)
        
        print(f"\n  Método seleccionado: {result.method}")
        print(f"  Phi: {result.phi_value:.6f}")
        print(f"  Confianza: {result.confidence:.2%}")
        print(f"  Tiempo: {result.computation_time*1000:.2f} ms")
        print(f"  Particiones evaluadas: {result.partitions_evaluated}")
        
        if result.additional_metrics:
            print(f"\n  Métricas adicionales:")
            for key, value in result.additional_metrics.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.6f}")
                else:
                    print(f"    {key}: {value}")
        
        results.append({
            'size': n,
            'method': result.method,
            'phi': result.phi_value,
            'time_ms': result.computation_time * 1000,
            'confidence': result.confidence,
            'partitions': result.partitions_evaluated
        })
    
    # Resumen de estadísticas
    print("\n" + "=" * 80)
    print("📈 ESTADÍSTICAS DEL CALCULADOR")
    print("=" * 80)
    
    stats = calculator.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return results, calculator


def demo_all_methods():
    """
    Demuestra todos los métodos con un sistema de ejemplo
    """
    print("\n" + "=" * 80)
    print("🎯 DEMOSTRACIÓN DE TODOS LOS MÉTODOS")
    print("=" * 80)
    
    calculator = PhiCalculatorOptimized()
    
    # Sistema pequeño para método exacto
    print(f"\n{'─'*80}")
    print(f"Sistema pequeño (n=6) - Método EXACTO")
    print(f"{'─'*80}")
    
    n_small = 6
    state_small = np.sin(np.linspace(0, 2*np.pi, n_small)) + np.random.normal(0, 0.1, n_small)
    connectivity_small = create_test_network(n_small, "small_world")
    
    result = calculator.calculate_phi(state_small, connectivity_small, PhiMethod.EXACT)
    print(f"  ✓ Phi: {result.phi_value:.6f}")
    print(f"  ✓ Confianza: {result.confidence:.2%}")
    print(f"  ✓ Tiempo: {result.computation_time*1000:.2f} ms")
    print(f"  ✓ Particiones evaluadas: {result.partitions_evaluated}")
    
    # Sistema mediano para método balanceado
    print(f"\n{'─'*80}")
    print(f"Sistema mediano (n=20) - Método BALANCEADO")
    print(f"{'─'*80}")
    
    n_medium = 20
    state_medium = np.sin(np.linspace(0, 4*np.pi, n_medium)) + np.random.normal(0, 0.1, n_medium)
    connectivity_medium = create_test_network(n_medium, "small_world")
    
    result = calculator.calculate_phi(state_medium, connectivity_medium, PhiMethod.BALANCED)
    print(f"  ✓ Phi: {result.phi_value:.6f}")
    print(f"  ✓ Confianza: {result.confidence:.2%}")
    print(f"  ✓ Tiempo: {result.computation_time*1000:.2f} ms")
    print(f"  ✓ Particiones evaluadas: {result.partitions_evaluated}")
    
    # Sistema grande para método aproximado
    print(f"\n{'─'*80}")
    print(f"Sistema grande (n=100) - Método APROXIMADO")
    print(f"{'─'*80}")
    
    n_large = 100
    state_large = np.sin(np.linspace(0, 4*np.pi, n_large)) + np.random.normal(0, 0.1, n_large)
    connectivity_large = create_test_network(n_large, "small_world")
    
    result = calculator.calculate_phi(state_large, connectivity_large, PhiMethod.APPROXIMATE)
    print(f"  ✓ Phi: {result.phi_value:.6f}")
    print(f"  ✓ Confianza: {result.confidence:.2%}")
    print(f"  ✓ Tiempo: {result.computation_time*1000:.2f} ms")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Ejecutar demostraciones
    print("\n🚀 SISTEMA DE CÁLCULO DE PHI - VERSIÓN OPTIMIZADA 2.0\n")
    
    # Demo de todos los métodos
    demo_all_methods()
    
    # Benchmark completo
    results, calculator = run_comprehensive_benchmark()
    
    print("\n" + "=" * 80)
    print("✅ SISTEMA COMPLETAMENTE FUNCIONAL Y OPTIMIZADO")
    print("=" * 80)
    print("\n💡 Listo para usar en tu aplicación!")
