# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai

"""
GUÍA DE USO: Sistema Optimizado de Cálculo de Phi
==================================================

Este documento muestra cómo usar el sistema optimizado en aplicaciones reales
"""

import numpy as np
import matplotlib.pyplot as plt
from phi_calculator_complete import (
    PhiCalculatorOptimized, 
    PhiMethod, 
    PhiResult,
    create_test_network
)


# ============================================================================
# EJEMPLO 1: Uso básico - Cálculo automático
# ============================================================================

def ejemplo_basico():
    """Ejemplo más simple posible"""
    print("=" * 80)
    print("EJEMPLO 1: Uso Básico")
    print("=" * 80)
    
    # 1. Crear calculador
    calculator = PhiCalculatorOptimized()
    
    # 2. Preparar datos
    n = 20
    state = np.random.randn(n)  # Estado aleatorio
    connectivity = create_test_network(n, "small_world")  # Red small-world
    
    # 3. Calcular Phi (automático)
    result = calculator.calculate_phi(state, connectivity)
    
    # 4. Usar resultado
    print(f"\nΦ = {result.phi_value:.6f}")
    print(f"Método usado: {result.method}")
    print(f"Confianza: {result.confidence:.1%}")
    print(f"Tiempo: {result.computation_time*1000:.2f} ms")
    
    return result


# ============================================================================
# EJEMPLO 2: Comparar diferentes estados
# ============================================================================

def ejemplo_comparacion_estados():
    """Compara Phi de diferentes tipos de estados"""
    print("\n" + "=" * 80)
    print("EJEMPLO 2: Comparación de Estados")
    print("=" * 80)
    
    calculator = PhiCalculatorOptimized()
    n = 30
    connectivity = create_test_network(n, "small_world")
    
    # Diferentes tipos de estados
    estados = {
        'Aleatorio': np.random.randn(n),
        'Estructurado (seno)': np.sin(np.linspace(0, 4*np.pi, n)),
        'Constante': np.ones(n),
        'Mixto': np.concatenate([np.ones(n//2), -np.ones(n//2)])
    }
    
    print("\nComparando Phi para diferentes estados:")
    print("-" * 80)
    
    resultados = {}
    for nombre, state in estados.items():
        result = calculator.calculate_phi(state, connectivity)
        resultados[nombre] = result
        
        print(f"\n{nombre:20s} | Φ = {result.phi_value:8.6f} | "
              f"Método: {result.method:12s} | "
              f"Tiempo: {result.computation_time*1000:6.2f} ms")
    
    # Encontrar el de mayor Phi
    max_phi_estado = max(resultados.items(), key=lambda x: x[1].phi_value)
    print(f"\n✓ Estado con mayor integración: {max_phi_estado[0]} "
          f"(Φ = {max_phi_estado[1].phi_value:.6f})")
    
    return resultados


# ============================================================================
# EJEMPLO 3: Análisis de series temporales
# ============================================================================

def ejemplo_serie_temporal():
    """Analiza cómo cambia Phi a lo largo del tiempo"""
    print("\n" + "=" * 80)
    print("EJEMPLO 3: Análisis de Serie Temporal")
    print("=" * 80)
    
    calculator = PhiCalculatorOptimized()
    n = 25
    connectivity = create_test_network(n, "small_world")
    
    # Simular evolución temporal
    n_timesteps = 20
    phi_evolution = []
    
    print("\nCalculando Phi en cada paso temporal...")
    
    for t in range(n_timesteps):
        # Estado que evoluciona con el tiempo
        frequency = 0.5 + 0.1 * t  # Frecuencia que aumenta
        state = np.sin(frequency * np.linspace(0, 2*np.pi, n))
        state += np.random.normal(0, 0.1, n)  # Ruido
        
        result = calculator.calculate_phi(state, connectivity)
        phi_evolution.append(result.phi_value)
        
        if t % 5 == 0:
            print(f"  t={t:2d}: Φ = {result.phi_value:.6f}")
    
    # Análisis de tendencia
    phi_array = np.array(phi_evolution)
    tendencia = "creciente" if np.mean(np.diff(phi_array)) > 0 else "decreciente"
    
    print(f"\nTendencia: {tendencia}")
    print(f"Φ promedio: {np.mean(phi_array):.6f}")
    print(f"Φ mínimo: {np.min(phi_array):.6f}")
    print(f"Φ máximo: {np.max(phi_array):.6f}")
    
    return phi_evolution


# ============================================================================
# EJEMPLO 4: Comparación de topologías de red
# ============================================================================

def ejemplo_topologias():
    """Compara Phi para diferentes topologías de red"""
    print("\n" + "=" * 80)
    print("EJEMPLO 4: Comparación de Topologías de Red")
    print("=" * 80)
    
    calculator = PhiCalculatorOptimized()
    n = 30
    state = np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 0.1, n)
    
    topologias = ['small_world', 'random', 'scale_free', 'ring']
    nombres_completos = {
        'small_world': 'Small-World (Watts-Strogatz)',
        'random': 'Aleatorio (Erdős-Rényi)',
        'scale_free': 'Libre de Escala (Barabási-Albert)',
        'ring': 'Anillo'
    }
    
    print("\nComparando Phi para diferentes topologías de red:")
    print("-" * 80)
    
    resultados = {}
    for topo in topologias:
        connectivity = create_test_network(n, topo)
        result = calculator.calculate_phi(state, connectivity)
        resultados[topo] = result
        
        nombre = nombres_completos[topo]
        print(f"\n{nombre:40s}")
        print(f"  Φ = {result.phi_value:.6f}")
        print(f"  Conectividad promedio = {np.mean(connectivity):.3f}")
    
    # Encontrar topología con mayor Phi
    max_topo = max(resultados.items(), key=lambda x: x[1].phi_value)
    print(f"\n✓ Topología con mayor integración: {nombres_completos[max_topo[0]]} "
          f"(Φ = {max_topo[1].phi_value:.6f})")
    
    return resultados


# ============================================================================
# EJEMPLO 5: Benchmark de rendimiento
# ============================================================================

def ejemplo_benchmark():
    """Benchmark de rendimiento en diferentes tamaños"""
    print("\n" + "=" * 80)
    print("EJEMPLO 5: Benchmark de Rendimiento")
    print("=" * 80)
    
    calculator = PhiCalculatorOptimized()
    
    tamaños = [5, 10, 20, 30, 50, 80, 100, 150, 200]
    tiempos = []
    metodos = []
    
    print("\nMidiendo tiempo de cálculo para diferentes tamaños:")
    print("-" * 80)
    print(f"{'Tamaño':>8} | {'Método':>12} | {'Tiempo (ms)':>12} | {'Speedup':>10}")
    print("-" * 80)
    
    tiempo_base = None
    
    for n in tamaños:
        state = np.random.randn(n)
        connectivity = create_test_network(n, "small_world")
        
        result = calculator.calculate_phi(state, connectivity)
        
        tiempo_ms = result.computation_time * 1000
        tiempos.append(tiempo_ms)
        metodos.append(result.method)
        
        if tiempo_base is None:
            tiempo_base = tiempo_ms
            speedup = 1.0
        else:
            # Speedup relativo al tiempo por elemento
            speedup = (tiempo_base / tamaños[0]) / (tiempo_ms / n)
        
        print(f"{n:8d} | {result.method:>12s} | {tiempo_ms:12.2f} | {speedup:10.2f}x")
    
    # Estadísticas finales
    print("\n" + "-" * 80)
    stats = calculator.get_stats()
    print(f"\nEstadísticas del calculador:")
    print(f"  Llamadas exactas: {stats['exact_calls']}")
    print(f"  Llamadas balanceadas: {stats['balanced_calls']}")
    print(f"  Llamadas aproximadas: {stats['approximate_calls']}")
    print(f"  Tiempo promedio: {stats['average_time']*1000:.2f} ms")
    
    return tiempos, metodos


# ============================================================================
# EJEMPLO 6: Uso con caché
# ============================================================================

def ejemplo_cache():
    """Demuestra el beneficio del caché"""
    print("\n" + "=" * 80)
    print("EJEMPLO 6: Beneficio del Caché")
    print("=" * 80)
    
    # Sin caché
    calculator_no_cache = PhiCalculatorOptimized(cache_enabled=False)
    
    # Con caché
    calculator_con_cache = PhiCalculatorOptimized(cache_enabled=True)
    
    n = 30
    state = np.random.randn(n)
    connectivity = create_test_network(n, "small_world")
    
    print("\nCalculando Phi 10 veces con los mismos datos:")
    print("-" * 80)
    
    # Sin caché
    tiempos_sin_cache = []
    for i in range(10):
        result = calculator_no_cache.calculate_phi(state, connectivity)
        tiempos_sin_cache.append(result.computation_time * 1000)
    
    # Con caché
    tiempos_con_cache = []
    for i in range(10):
        result = calculator_con_cache.calculate_phi(state, connectivity)
        tiempos_con_cache.append(result.computation_time * 1000)
    
    print(f"\nSin caché:")
    print(f"  Tiempo promedio: {np.mean(tiempos_sin_cache):.2f} ms")
    print(f"  Tiempo total: {np.sum(tiempos_sin_cache):.2f} ms")
    
    print(f"\nCon caché:")
    print(f"  Tiempo promedio: {np.mean(tiempos_con_cache):.2f} ms")
    print(f"  Tiempo total: {np.sum(tiempos_con_cache):.2f} ms")
    print(f"  Cache hits: {calculator_con_cache.stats['cache_hits']}")
    
    speedup = np.mean(tiempos_sin_cache) / np.mean(tiempos_con_cache)
    print(f"\n✓ Aceleración con caché: {speedup:.1f}x")
    
    return speedup


# ============================================================================
# EJEMPLO 7: Selección manual de método
# ============================================================================

def ejemplo_seleccion_manual():
    """Demuestra cómo seleccionar método manualmente"""
    print("\n" + "=" * 80)
    print("EJEMPLO 7: Selección Manual de Método")
    print("=" * 80)
    
    calculator = PhiCalculatorOptimized()
    n = 30
    state = np.random.randn(n)
    connectivity = create_test_network(n, "small_world")
    
    print("\nCalculando con diferentes métodos (mismo estado):")
    print("-" * 80)
    
    metodos = [
        (PhiMethod.BALANCED, "Balanceado (recomendado)"),
        (PhiMethod.APPROXIMATE, "Aproximado (rápido)")
    ]
    
    for metodo, nombre in metodos:
        result = calculator.calculate_phi(state, connectivity, method=metodo)
        
        print(f"\n{nombre}:")
        print(f"  Φ = {result.phi_value:.6f}")
        print(f"  Confianza = {result.confidence:.1%}")
        print(f"  Tiempo = {result.computation_time*1000:.2f} ms")
        print(f"  Particiones evaluadas = {result.partitions_evaluated}")


# ============================================================================
# EJEMPLO 8: Análisis de mejor partición
# ============================================================================

def ejemplo_mejor_particion():
    """Analiza la mejor partición encontrada"""
    print("\n" + "=" * 80)
    print("EJEMPLO 8: Análisis de la Mejor Partición")
    print("=" * 80)
    
    calculator = PhiCalculatorOptimized()
    n = 16  # Tamaño pequeño para visualización
    state = np.sin(np.linspace(0, 4*np.pi, n))
    connectivity = create_test_network(n, "small_world")
    
    result = calculator.calculate_phi(state, connectivity, PhiMethod.BALANCED)
    
    print(f"\nΦ = {result.phi_value:.6f}")
    print(f"Mejor partición encontrada:")
    
    if result.best_partition is not None:
        part_a = np.where(result.best_partition == 0)[0]
        part_b = np.where(result.best_partition == 1)[0]
        
        print(f"\n  Partición A (0): {len(part_a)} nodos")
        print(f"    Nodos: {list(part_a)}")
        print(f"\n  Partición B (1): {len(part_b)} nodos")
        print(f"    Nodos: {list(part_b)}")
        
        # Analizar estados en cada partición
        state_a = state[part_a]
        state_b = state[part_b]
        
        print(f"\n  Estadísticas:")
        print(f"    Media A: {np.mean(state_a):.3f}, Media B: {np.mean(state_b):.3f}")
        print(f"    Std A: {np.std(state_a):.3f}, Std B: {np.std(state_b):.3f}")
    
    return result


# ============================================================================
# FUNCIÓN PRINCIPAL - EJECUTAR TODOS LOS EJEMPLOS
# ============================================================================

def ejecutar_todos_ejemplos():
    """Ejecuta todos los ejemplos de uso"""
    print("\n" + "🚀" * 40)
    print("GUÍA COMPLETA DE USO - SISTEMA OPTIMIZADO DE PHI")
    print("🚀" * 40 + "\n")
    
    ejemplos = [
        ("Uso Básico", ejemplo_basico),
        ("Comparación de Estados", ejemplo_comparacion_estados),
        ("Serie Temporal", ejemplo_serie_temporal),
        ("Topologías de Red", ejemplo_topologias),
        ("Benchmark", ejemplo_benchmark),
        ("Caché", ejemplo_cache),
        ("Selección Manual", ejemplo_seleccion_manual),
        ("Mejor Partición", ejemplo_mejor_particion)
    ]
    
    for i, (nombre, funcion) in enumerate(ejemplos, 1):
        try:
            print(f"\n{'▶'*3} Ejecutando ejemplo {i}/{len(ejemplos)}: {nombre}")
            funcion()
            print(f"\n{'✓'*3} Ejemplo {i} completado exitosamente\n")
        except Exception as e:
            print(f"\n{'✗'*3} Error en ejemplo {i}: {e}\n")
    
    print("\n" + "=" * 80)
    print("✅ TODOS LOS EJEMPLOS COMPLETADOS")
    print("=" * 80)
    print("\n💡 El sistema está listo para usar en tu aplicación!")
    print("💡 Consulta el código fuente para más detalles de implementación\n")


# ============================================================================
# RESUMEN DE API
# ============================================================================

def mostrar_api():
    """Muestra resumen de la API disponible"""
    print("\n" + "=" * 80)
    print("📚 RESUMEN DE API")
    print("=" * 80)
    
    api_doc = """
    
CLASE PRINCIPAL: PhiCalculatorOptimized
---------------------------------------

Inicialización:
    calculator = PhiCalculatorOptimized(
        cache_enabled=True,      # Usar caché para acelerar
        max_cache_size=1000      # Tamaño máximo del caché
    )

Método principal:
    result = calculator.calculate_phi(
        state,                   # np.ndarray: Vector de estado (n elementos)
        connectivity,            # np.ndarray: Matriz conectividad (n×n)
        method=PhiMethod.AUTO    # AUTO, EXACT, BALANCED, o APPROXIMATE
    )

Resultado (PhiResult):
    result.phi_value            # float: Valor de Φ
    result.method               # str: Método usado
    result.confidence           # float: Confianza (0-1)
    result.computation_time     # float: Tiempo en segundos
    result.best_partition       # np.ndarray: Mejor partición encontrada
    result.partitions_evaluated # int: Particiones evaluadas
    result.additional_metrics   # dict: Métricas adicionales

Métodos auxiliares:
    calculator.get_stats()      # Obtener estadísticas de uso
    calculator.clear_cache()    # Limpiar caché
    calculator.reset_stats()    # Reiniciar estadísticas

FUNCIONES AUXILIARES:
--------------------

create_test_network(n, network_type):
    Crea red de prueba
    network_type: "small_world", "random", "scale_free", "ring"

MÉTODOS DISPONIBLES (PhiMethod):
--------------------------------

PhiMethod.AUTO:         Selección automática según tamaño
PhiMethod.EXACT:        Cálculo exacto (n ≤ 8)
PhiMethod.BALANCED:     Balanceado (8 < n ≤ 80)
PhiMethod.APPROXIMATE:  Aproximado (n > 80)

RECOMENDACIONES DE USO:
----------------------

1. Usa PhiMethod.AUTO para la mayoría de casos
2. Habilita caché si calculas Phi repetidamente
3. Para n > 100, el método aproximado es la única opción práctica
4. Para precisión máxima con n ≤ 8, usa método exacto
5. Método balanceado es el mejor compromiso para 10 ≤ n ≤ 80

RANGOS DE RENDIMIENTO:
---------------------

n ≤ 8:      < 20 ms     (método exacto)
8 < n ≤ 80: < 150 ms    (método balanceado)
n > 80:     < 10 ms     (método aproximado)

    """
    print(api_doc)


if __name__ == "__main__":
    # Mostrar API
    mostrar_api()
    
    # Ejecutar todos los ejemplos
    ejecutar_todos_ejemplos()
