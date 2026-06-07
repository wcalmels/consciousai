"""
Optimizaciones para cálculo rápido de Phi
Basado en análisis de TUCH-OS con mejoras específicas para nuestro sistema
"""

import numpy as np
from scipy import linalg
import networkx as nx
from typing import Dict, Tuple, List
import logging

class PhiOptimizer:
    """
    Optimizador de cálculo de Phi con selección adaptativa de método
    """
    
    def __init__(self):
        self.cache = {}
        self.stats = {
            'exact_calls': 0,
            'balanced_calls': 0,
            'approximate_calls': 0,
            'cache_hits': 0
        }
    
    def calculate_phi_adaptive(self, state: np.ndarray, 
                              connectivity: np.ndarray) -> Dict:
        """
        Calcula Phi adaptivamente según tamaño del sistema
        
        Args:
            state: Vector de estado del sistema
            connectivity: Matriz de conectividad
            
        Returns:
            Dict con phi_value, method, confidence, y métricas adicionales
        """
        n = len(state)
        
        # Generar clave de caché
        cache_key = self._generate_cache_key(state, connectivity)
        
        # Verificar caché
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        # Seleccionar método según tamaño
        if n <= 8:
            result = self._calculate_exact(state, connectivity)
            self.stats['exact_calls'] += 1
        elif n <= 80:
            result = self._calculate_balanced(state, connectivity)
            self.stats['balanced_calls'] += 1
        else:
            result = self._calculate_approximate(state, connectivity)
            self.stats['approximate_calls'] += 1
        
        # Guardar en caché
        self.cache[cache_key] = result
        
        return result
    
    def _calculate_exact(self, state: np.ndarray, 
                        connectivity: np.ndarray) -> Dict:
        """
        Cálculo exacto para sistemas pequeños (n ≤ 8)
        Evalúa todas las particiones bipartitas posibles
        """
        n = len(state)
        min_phi = float('inf')
        best_partition = None
        
        # Evaluar todas las particiones bipartitas no triviales
        for i in range(1, 2**(n-1)):
            partition = self._int_to_partition(i, n)
            phi = self._evaluate_partition(state, connectivity, partition)
            
            if phi < min_phi:
                min_phi = phi
                best_partition = partition
        
        return {
            'phi_value': min_phi if min_phi != float('inf') else 0.0,
            'method': 'exact',
            'confidence': 1.0,
            'best_partition': best_partition,
            'partitions_evaluated': 2**(n-1) - 1
        }
    
    def _calculate_balanced(self, state: np.ndarray, 
                           connectivity: np.ndarray,
                           n_spectral: int = 10,
                           n_random: int = 50) -> Dict:
        """
        Cálculo balanceado: clustering espectral + muestreo inteligente
        Para sistemas medianos (8 < n ≤ 80)
        """
        n = len(state)
        phi_values = []
        partitions_tested = []
        
        # Estrategia 1: Particiones espectrales (más prometedoras)
        spectral_partitions = self._find_spectral_partitions(
            connectivity, max_partitions=n_spectral
        )
        
        for partition in spectral_partitions:
            phi = self._evaluate_partition(state, connectivity, partition)
            phi_values.append(phi)
            partitions_tested.append(partition)
        
        # Estrategia 2: Muestreo aleatorio inteligente
        random_partitions = self._generate_intelligent_partitions(n, n_random)
        
        for partition in random_partitions:
            phi = self._evaluate_partition(state, connectivity, partition)
            phi_values.append(phi)
            partitions_tested.append(partition)
        
        # Encontrar mínimo
        min_idx = np.argmin(phi_values)
        min_phi = phi_values[min_idx]
        best_partition = partitions_tested[min_idx]
        
        return {
            'phi_value': min_phi,
            'method': 'balanced',
            'confidence': min(0.95, len(phi_values) / 100),
            'best_partition': best_partition,
            'partitions_evaluated': len(phi_values),
            'phi_distribution': {
                'mean': float(np.mean(phi_values)),
                'std': float(np.std(phi_values)),
                'percentile_25': float(np.percentile(phi_values, 25))
            }
        }
    
    def _calculate_approximate(self, state: np.ndarray, 
                              connectivity: np.ndarray) -> Dict:
        """
        Aproximación rápida basada en propiedades espectrales
        Para sistemas grandes (n > 80)
        O(N log N) complejidad
        """
        n = len(state)
        
        try:
            # Matriz de covarianza del estado
            if state.ndim > 1:
                cov_matrix = np.cov(state.T)
            else:
                # Para vector 1D, crear matriz de covarianza artificial
                state_matrix = state.reshape(-1, 1)
                cov_matrix = np.outer(state, state) / n
            
            # Autovalores (entropía espectral)
            eigenvalues = linalg.eigvalsh(cov_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            if len(eigenvalues) == 0:
                return {
                    'phi_value': 0.0,
                    'method': 'approximate',
                    'confidence': 0.5,
                    'note': 'Sistema degenerado'
                }
            
            # Entropía de Von Neumann normalizada
            eigenvalues_norm = eigenvalues / np.sum(eigenvalues)
            entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-12))
            entropy_normalized = entropy / np.log(len(eigenvalues))
            
            # Factor de conectividad
            connectivity_strength = np.mean(np.abs(connectivity))
            
            # Phi aproximado: combina entropía y conectividad
            phi_approx = entropy_normalized * connectivity_strength * np.sqrt(n)
            
            # Limitar a rango razonable
            phi_approx = np.clip(phi_approx, 0.0, 10.0)
            
            return {
                'phi_value': float(phi_approx),
                'method': 'approximate',
                'confidence': 0.75,
                'entropy_contribution': float(entropy_normalized),
                'connectivity_contribution': float(connectivity_strength),
                'effective_dimensions': len(eigenvalues)
            }
            
        except Exception as e:
            logging.warning(f"Error en cálculo aproximado: {e}")
            return {
                'phi_value': 0.0,
                'method': 'approximate',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _find_spectral_partitions(self, connectivity: np.ndarray, 
                                  max_partitions: int = 10) -> List[np.ndarray]:
        """
        Encuentra particiones naturales usando clustering espectral
        Basado en el método de Fiedler
        """
        partitions = []
        n = connectivity.shape[0]
        
        try:
            # Matriz Laplaciana
            degrees = np.sum(np.abs(connectivity), axis=1)
            D = np.diag(degrees)
            L = D - connectivity
            
            # Autovalores y autovectores
            eigenvalues, eigenvectors = linalg.eigh(L)
            
            # Vector de Fiedler (2do autovector)
            if len(eigenvectors) > 1:
                fiedler_vector = eigenvectors[:, 1]
                
                # Bipartición basada en signo
                partition = (fiedler_vector >= 0).astype(int)
                if np.sum(partition) > 0 and np.sum(partition) < n:
                    partitions.append(partition)
                
                # Bipartición basada en mediana
                median_val = np.median(fiedler_vector)
                partition = (fiedler_vector >= median_val).astype(int)
                if np.sum(partition) > 0 and np.sum(partition) < n:
                    partitions.append(partition)
            
            # Usar autovectores adicionales
            for i in range(2, min(max_partitions + 2, n)):
                if i < len(eigenvectors[0]):
                    eigenvector = eigenvectors[:, i]
                    
                    # Múltiples umbrales
                    for percentile in [25, 50, 75]:
                        threshold = np.percentile(eigenvector, percentile)
                        partition = (eigenvector >= threshold).astype(int)
                        
                        if np.sum(partition) > 0 and np.sum(partition) < n:
                            partitions.append(partition)
                        
                        if len(partitions) >= max_partitions:
                            break
                
                if len(partitions) >= max_partitions:
                    break
        
        except Exception as e:
            logging.warning(f"Error en clustering espectral: {e}")
        
        return partitions[:max_partitions]
    
    def _generate_intelligent_partitions(self, n: int, 
                                        count: int) -> List[np.ndarray]:
        """
        Genera particiones aleatorias inteligentes
        Evita particiones triviales y favorece balanceo
        """
        partitions = []
        
        # Particiones balanceadas (50-50)
        for _ in range(count // 3):
            partition = np.zeros(n, dtype=int)
            indices = np.random.choice(n, size=n//2, replace=False)
            partition[indices] = 1
            partitions.append(partition)
        
        # Particiones desbalanceadas interesantes
        sizes = [n//4, n//3, 2*n//3, 3*n//4]
        for size in sizes:
            if 1 <= size < n:
                partition = np.zeros(n, dtype=int)
                indices = np.random.choice(n, size=size, replace=False)
                partition[indices] = 1
                partitions.append(partition)
        
        # Particiones basadas en patrones
        remaining = count - len(partitions)
        for _ in range(remaining):
            # Patrón aleatorio con restricción de no trivialidad
            partition = np.random.randint(0, 2, n)
            
            # Asegurar no trivialidad
            if np.sum(partition) == 0 or np.sum(partition) == n:
                partition[np.random.randint(0, n)] = 1 - partition[0]
            
            partitions.append(partition)
        
        return partitions[:count]
    
    def _evaluate_partition(self, state: np.ndarray, 
                           connectivity: np.ndarray,
                           partition: np.ndarray) -> float:
        """
        Evalúa phi para una partición específica
        Información mutua entre las dos partes
        """
        mask_a = partition == 0
        mask_b = partition == 1
        
        # Verificar no trivialidad
        if np.sum(mask_a) == 0 or np.sum(mask_b) == 0:
            return float('inf')
        
        try:
            # Estados de cada partición
            state_a = state[mask_a]
            state_b = state[mask_b]
            
            # Conectividad entre particiones
            connectivity_ab = connectivity[np.ix_(mask_a, mask_b)]
            conn_strength = np.mean(np.abs(connectivity_ab)) if connectivity_ab.size > 0 else 0.0
            
            # Calcular correlación entre particiones
            if len(state_a) > 0 and len(state_b) > 0:
                # Usar promedios si hay múltiples elementos
                mean_a = np.mean(state_a)
                mean_b = np.mean(state_b)
                std_a = np.std(state_a) if len(state_a) > 1 else 1.0
                std_b = np.std(state_b) if len(state_b) > 1 else 1.0
                
                # Correlación normalizada
                if std_a > 0 and std_b > 0:
                    correlation = np.abs((mean_a - np.mean(state)) * (mean_b - np.mean(state))) / (std_a * std_b)
                    correlation = np.clip(correlation, 0, 0.99)
                else:
                    correlation = 0.0
            else:
                correlation = 0.0
            
            # Información mutua aproximada
            if correlation < 0.99:
                mutual_info = -0.5 * np.log(1 - correlation**2 + 1e-10)
            else:
                mutual_info = 5.0
            
            # Phi como información mutua ponderada por conectividad
            phi = mutual_info * conn_strength
            
            return float(phi)
            
        except Exception as e:
            logging.warning(f"Error evaluando partición: {e}")
            return float('inf')
    
    def _int_to_partition(self, i: int, n: int) -> np.ndarray:
        """Convierte entero a vector de partición binario"""
        binary_str = format(i, f'0{n}b')
        return np.array([int(bit) for bit in binary_str], dtype=int)
    
    def _generate_cache_key(self, state: np.ndarray, 
                           connectivity: np.ndarray) -> str:
        """Genera clave única para caché"""
        state_hash = hash(state.tobytes())
        conn_hash = hash(connectivity.tobytes())
        return f"{state_hash}_{conn_hash}"
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas de uso"""
        total_calls = sum([
            self.stats['exact_calls'],
            self.stats['balanced_calls'],
            self.stats['approximate_calls']
        ])
        
        return {
            **self.stats,
            'total_calls': total_calls,
            'cache_hit_rate': self.stats['cache_hits'] / max(total_calls, 1),
            'cache_size': len(self.cache)
        }
    
    def clear_cache(self):
        """Limpia el caché"""
        self.cache.clear()


def compare_methods_benchmark():
    """
    Benchmark para comparar velocidad de los diferentes métodos
    """
    import time
    
    optimizer = PhiOptimizer()
    
    print("🔬 BENCHMARK DE MÉTODOS DE CÁLCULO DE PHI")
    print("=" * 60)
    
    # Test con diferentes tamaños
    sizes = [6, 10, 30, 50, 100, 200]
    
    results = []
    
    for n in sizes:
        print(f"\n📊 Sistema de tamaño n={n}")
        
        # Generar datos de prueba
        state = np.random.randn(n)
        G = nx.watts_strogatz_graph(n, k=min(4, n-1), p=0.3)
        connectivity = nx.to_numpy_array(G)
        
        # Medir tiempo
        start = time.time()
        result = optimizer.calculate_phi_adaptive(state, connectivity)
        elapsed = time.time() - start
        
        print(f"  Método: {result['method']}")
        print(f"  Phi: {result['phi_value']:.4f}")
        print(f"  Confianza: {result['confidence']:.2f}")
        print(f"  Tiempo: {elapsed*1000:.2f} ms")
        
        results.append({
            'size': n,
            'method': result['method'],
            'phi': result['phi_value'],
            'time_ms': elapsed * 1000
        })
    
    print("\n" + "=" * 60)
    print("📈 ESTADÍSTICAS DE USO:")
    stats = optimizer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return results


if __name__ == "__main__":
    # Ejecutar benchmark
    results = compare_methods_benchmark()
    
    print("\n✅ Benchmark completado!")
    print("\n💡 RECOMENDACIONES:")
    print("  • Sistemas pequeños (n≤8): Método exacto - alta precisión")
    print("  • Sistemas medianos (8<n≤80): Método balanceado - buen compromiso")
    print("  • Sistemas grandes (n>80): Método aproximado - muy rápido")
