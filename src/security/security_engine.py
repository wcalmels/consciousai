# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai

"""
ConsciousAI Security Module v2.0 - Production Grade
=====================================================

Incorpora conceptos de TUCH-OS Security con implementación robusta:
- Topological Protection (hash chain)
- Adaptive Monitoring (metacognition)
- File Integrity Checking
- Non-destructive Security

Author: Walter Calmels - ConsciousAI Platform
Version: 2.0.0-Security
Based on: TUCH-OS Security concepts (rewritten for production)
"""

import hashlib
import time
import os
import sys
import logging
from typing import List, Dict, Optional, Tuple, Callable, Any
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import threading
import json
from pathlib import Path

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available - topological protection disabled")

import numpy as np

logger = logging.getLogger('ConsciousAI_Security')

# =============================================================================
# CONFIGURATION
# =============================================================================

class SecurityMode(IntEnum):
    """Security modes"""
    DEVELOPMENT = 0  # Minimal security, allow debugging
    STAGING = 1      # Moderate security, warnings only
    PRODUCTION = 2   # Full security, strict enforcement

@dataclass
class SecurityConfig:
    """Security configuration"""
    mode: SecurityMode = SecurityMode.DEVELOPMENT
    monitoring_interval: float = 30.0
    enable_file_monitoring: bool = True
    enable_topological_protection: bool = True
    enable_adaptive_monitoring: bool = True
    protected_files: List[str] = None
    alert_on_violations: bool = True
    
    def __post_init__(self):
        if self.protected_files is None:
            self.protected_files = []

# =============================================================================
# TOPOLOGICAL PROTECTION (Hash Chain)
# =============================================================================

class TopologicalProtection:
    """
    Protección mediante hash chain topológico.
    
    Cada capa del sistema tiene un hash que depende de:
    - Su propia firma (código/config)
    - Hash de la capa anterior
    - Salt aleatorio
    
    Cualquier modificación rompe la cadena y es detectable.
    """
    
    def __init__(self, layer_definitions: Dict[str, str]):
        """
        Initialize topological protection.
        
        Args:
            layer_definitions: Dict mapping layer_name -> layer_signature
                              (en producción: hash del bytecode real)
        """
        if not NETWORKX_AVAILABLE:
            raise RuntimeError("NetworkX required for topological protection")
        
        self.graph = nx.DiGraph()
        self.layer_signatures = layer_definitions
        self.layer_hashes = {}
        self.salt = os.urandom(32).hex()
        self.last_verification = time.time()
        
        self._build_graph()
        self._initialize_hashes()
        
        logger.info(f"Topological protection initialized ({len(layer_definitions)} layers)")
    
    def _build_graph(self):
        """Construye grafo de dependencias entre capas"""
        layers = list(self.layer_signatures.keys())
        
        # Crear grafo dirigido: cada capa depende de la anterior
        for i in range(len(layers) - 1):
            self.graph.add_edge(layers[i], layers[i+1])
    
    def _initialize_hashes(self):
        """Inicializa hash chain para todas las capas"""
        topo_order = list(nx.topological_sort(self.graph))
        prev_hash = hashlib.sha256(b"GENESIS_BLOCK").hexdigest()
        
        for layer in topo_order:
            signature = self.layer_signatures[layer]
            combined = f"{signature}{prev_hash}{self.salt}".encode()
            current_hash = hashlib.sha256(combined).hexdigest()
            
            self.layer_hashes[layer] = {
                'hash': current_hash,
                'signature': signature,
                'prev_hash': prev_hash,
                'timestamp': time.time()
            }
            
            prev_hash = current_hash
    
    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verifica integridad de todas las capas.
        
        Returns:
            (is_valid, tampered_layer): True si integridad OK, None si válido
                                       False + nombre de capa si tampering
        """
        topo_order = list(nx.topological_sort(self.graph))
        prev_hash = hashlib.sha256(b"GENESIS_BLOCK").hexdigest()
        
        for layer in topo_order:
            stored_info = self.layer_hashes[layer]
            signature = self.layer_signatures[layer]
            
            # Recalcular hash esperado
            combined = f"{signature}{prev_hash}{self.salt}".encode()
            expected_hash = hashlib.sha256(combined).hexdigest()
            
            # Comparar con hash almacenado
            if expected_hash != stored_info['hash']:
                logger.critical(
                    f"Topological tampering detected in layer '{layer}'\n"
                    f"Expected: {expected_hash}\n"
                    f"Found: {stored_info['hash']}"
                )
                return False, layer
            
            prev_hash = stored_info['hash']
        
        self.last_verification = time.time()
        return True, None
    
    def update_layer(self, layer: str, new_signature: str):
        """
        Actualiza legítimamente la firma de una capa.
        
        Args:
            layer: Nombre de la capa
            new_signature: Nueva firma (e.g., hash del código actualizado)
        """
        if layer not in self.layer_signatures:
            raise ValueError(f"Unknown layer: {layer}")
        
        logger.info(f"Updating layer '{layer}' signature")
        self.layer_signatures[layer] = new_signature
        self._initialize_hashes()  # Rebuild entire hash chain
    
    def get_layer_info(self, layer: str) -> Dict:
        """Obtiene información de una capa"""
        if layer not in self.layer_hashes:
            return {}
        return self.layer_hashes[layer].copy()

# =============================================================================
# FILE INTEGRITY MONITORING
# =============================================================================

class FileIntegrityMonitor:
    """
    Monitoreo de integridad de archivos críticos.
    
    Crea baseline de archivos protegidos y detecta modificaciones.
    NO intenta reparar/restaurar automáticamente (requiere decisión humana).
    """
    
    def __init__(self, protected_paths: List[str]):
        """
        Initialize file monitor.
        
        Args:
            protected_paths: Lista de rutas de archivos a proteger
        """
        self.baselines = {}
        self.violations_history = deque(maxlen=100)
        
        for path in protected_paths:
            if os.path.exists(path):
                try:
                    self.baselines[path] = self._create_baseline(path)
                    logger.info(f"Baseline created for: {path}")
                except Exception as e:
                    logger.error(f"Failed to create baseline for {path}: {e}")
    
    def _create_baseline(self, path: str) -> Dict:
        """Crea baseline de archivo"""
        stat = os.stat(path)
        
        return {
            'hash': self._calculate_hash(path),
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'ctime': stat.st_ctime,
            'created_at': time.time()
        }
    
    def _calculate_hash(self, path: str) -> str:
        """Calcula hash SHA-256 de archivo"""
        sha256 = hashlib.sha256()
        
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {path}: {e}")
            return "ERROR"
    
    def verify_all(self) -> List[Dict]:
        """
        Verifica todos los archivos protegidos.
        
        Returns:
            Lista de violaciones detectadas (empty si todo OK)
        """
        violations = []
        
        for path, baseline in self.baselines.items():
            # Check existence
            if not os.path.exists(path):
                violation = {
                    'path': path,
                    'type': 'DELETED',
                    'severity': 10,
                    'timestamp': time.time()
                }
                violations.append(violation)
                self.violations_history.append(violation)
                continue
            
            # Check hash (most reliable)
            current_hash = self._calculate_hash(path)
            if current_hash != baseline['hash']:
                violation = {
                    'path': path,
                    'type': 'MODIFIED',
                    'severity': 9,
                    'baseline_hash': baseline['hash'],
                    'current_hash': current_hash,
                    'timestamp': time.time()
                }
                violations.append(violation)
                self.violations_history.append(violation)
                continue
            
            # Check size (fast secondary check)
            current_size = os.path.getsize(path)
            if current_size != baseline['size']:
                violation = {
                    'path': path,
                    'type': 'SIZE_CHANGED',
                    'severity': 7,
                    'baseline_size': baseline['size'],
                    'current_size': current_size,
                    'timestamp': time.time()
                }
                violations.append(violation)
                self.violations_history.append(violation)
        
        return violations
    
    def update_baseline(self, path: str):
        """Actualiza baseline de un archivo (después de cambio legítimo)"""
        if path not in self.baselines:
            raise ValueError(f"Path not monitored: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        logger.info(f"Updating baseline for: {path}")
        self.baselines[path] = self._create_baseline(path)
    
    def get_violation_history(self, last_n: int = 10) -> List[Dict]:
        """Obtiene últimas N violaciones"""
        return list(self.violations_history)[-last_n:]

# =============================================================================
# ADAPTIVE MONITORING (Metacognition)
# =============================================================================

class AdaptiveSecurityMonitor:
    """
    Monitor adaptativo que ajusta parámetros basado en efectividad.
    
    Usa "metacognición" para evaluar si las contramedidas funcionan
    y ajusta sensibilidad dinámicamente.
    """
    
    def __init__(self, initial_sensitivity: float = 0.5):
        """
        Initialize adaptive monitor.
        
        Args:
            initial_sensitivity: Sensibilidad inicial (0-1)
        """
        self.threat_history = deque(maxlen=100)
        self.repair_effectiveness_history = deque(maxlen=50)
        self.current_sensitivity = initial_sensitivity
        self.monitoring_interval = 30.0
        self.alert_threshold = 3
        
        logger.info(f"Adaptive monitoring initialized (sensitivity: {initial_sensitivity})")
    
    def record_threat(self, threat_type: str, severity: int, details: Dict = None):
        """
        Registra amenaza detectada.
        
        Args:
            threat_type: Tipo de amenaza (e.g., 'FILE_MODIFIED', 'TAMPERING')
            severity: Gravedad 1-10
            details: Información adicional
        """
        threat = {
            'timestamp': time.time(),
            'type': threat_type,
            'severity': severity,
            'details': details or {}
        }
        
        self.threat_history.append(threat)
        
        logger.warning(
            f"Threat recorded: {threat_type} (severity: {severity})"
        )
    
    def evaluate_repair_effectiveness(
        self,
        threats_before: int,
        threats_after: int,
        repair_type: str
    ) -> float:
        """
        Evalúa efectividad de una reparación (metacognición).
        
        Args:
            threats_before: Número de amenazas antes de repair
            threats_after: Número de amenazas después de repair
            repair_type: Tipo de reparación realizada
        
        Returns:
            Efectividad (0-1, donde 1 = muy efectivo)
        """
        reduction = max(0, threats_before - threats_after)
        effectiveness = reduction / max(1, threats_before)
        
        record = {
            'repair_type': repair_type,
            'effectiveness': effectiveness,
            'threats_before': threats_before,
            'threats_after': threats_after,
            'timestamp': time.time()
        }
        
        self.repair_effectiveness_history.append(record)
        
        logger.info(
            f"Repair effectiveness: {effectiveness:.2%} "
            f"({repair_type}: {threats_before} -> {threats_after})"
        )
        
        return effectiveness
    
    def adapt_parameters(self):
        """
        Ajusta parámetros dinámicamente basado en efectividad histórica.
        
        Esta es la "metacognición": El sistema evalúa su propio desempeño
        y se auto-ajusta para mejorar.
        """
        if len(self.repair_effectiveness_history) < 10:
            return  # Not enough data
        
        recent_repairs = list(self.repair_effectiveness_history)[-10:]
        avg_effectiveness = np.mean([r['effectiveness'] for r in recent_repairs])
        
        # Low effectiveness -> Increase sensitivity
        if avg_effectiveness < 0.3:
            old_sensitivity = self.current_sensitivity
            old_interval = self.monitoring_interval
            
            self.current_sensitivity = min(1.0, self.current_sensitivity + 0.1)
            self.monitoring_interval *= 0.8  # Check more frequently
            self.alert_threshold = max(1, self.alert_threshold - 1)
            
            logger.warning(
                f"Metacognition: Low repair effectiveness ({avg_effectiveness:.2%})\n"
                f"  Sensitivity: {old_sensitivity:.2f} -> {self.current_sensitivity:.2f}\n"
                f"  Interval: {old_interval:.1f}s -> {self.monitoring_interval:.1f}s"
            )
        
        # High effectiveness -> Can relax slightly
        elif avg_effectiveness > 0.8:
            old_sensitivity = self.current_sensitivity
            old_interval = self.monitoring_interval
            
            self.current_sensitivity = max(0.2, self.current_sensitivity - 0.05)
            self.monitoring_interval = min(60.0, self.monitoring_interval * 1.1)
            
            logger.info(
                f"Metacognition: High repair effectiveness ({avg_effectiveness:.2%})\n"
                f"  Sensitivity: {old_sensitivity:.2f} -> {self.current_sensitivity:.2f}\n"
                f"  Interval: {old_interval:.1f}s -> {self.monitoring_interval:.1f}s"
            )
    
    def should_trigger_alert(self) -> bool:
        """Determina si se debe activar alerta basado en threat rate"""
        if len(self.threat_history) < 5:
            return False
        
        # Threats en último minuto
        recent_threats = [
            t for t in self.threat_history
            if time.time() - t['timestamp'] < 60.0
        ]
        
        threat_rate = len(recent_threats)  # threats/minute
        threshold = self.alert_threshold * self.current_sensitivity
        
        return threat_rate > threshold
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del monitor"""
        if not self.threat_history:
            return {
                'total_threats': 0,
                'threat_rate': 0.0,
                'avg_severity': 0.0,
                'current_sensitivity': self.current_sensitivity,
                'monitoring_interval': self.monitoring_interval
            }
        
        recent_threats = [
            t for t in self.threat_history
            if time.time() - t['timestamp'] < 300.0  # Last 5 minutes
        ]
        
        return {
            'total_threats': len(self.threat_history),
            'recent_threats': len(recent_threats),
            'threat_rate': len(recent_threats) / 5.0,  # per minute
            'avg_severity': np.mean([t['severity'] for t in self.threat_history]),
            'current_sensitivity': self.current_sensitivity,
            'monitoring_interval': self.monitoring_interval,
            'alert_threshold': self.alert_threshold
        }

# =============================================================================
# MAIN SECURITY MODULE
# =============================================================================

class ConsciousAISecurityModule:
    """
    Módulo de seguridad principal para ConsciousAI.
    
    Integra:
    - Topological protection (hash chain)
    - File integrity monitoring
    - Adaptive monitoring (metacognition)
    - Non-destructive security (no sys.exit(), no process killing)
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize security module.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Initialize components
        self.topo_protection = None
        self.file_monitor = None
        self.adaptive_monitor = None
        
        self._initialize_components()
        
        logger.info(
            f"ConsciousAI Security Module initialized (mode: {config.mode.name})"
        )
    
    def _initialize_components(self):
        """Inicializa componentes de seguridad"""
        
        # Topological protection
        if self.config.enable_topological_protection and NETWORKX_AVAILABLE:
            layer_sigs = self._get_layer_signatures()
            self.topo_protection = TopologicalProtection(layer_sigs)
        
        # File monitoring
        if self.config.enable_file_monitoring and self.config.protected_files:
            self.file_monitor = FileIntegrityMonitor(self.config.protected_files)
        
        # Adaptive monitoring
        if self.config.enable_adaptive_monitoring:
            initial_sensitivity = {
                SecurityMode.DEVELOPMENT: 0.2,
                SecurityMode.STAGING: 0.5,
                SecurityMode.PRODUCTION: 0.8
            }[self.config.mode]
            
            self.adaptive_monitor = AdaptiveSecurityMonitor(initial_sensitivity)
    
    def _get_layer_signatures(self) -> Dict[str, str]:
        """
        Obtiene firmas de capas del sistema.
        
        En producción: Hash real del bytecode de cada capa.
        Aquí: Placeholders para demo.
        """
        return {
            'input_validation': 'input_layer_v1_' + os.urandom(8).hex(),
            'preprocessing': 'preproc_layer_v1_' + os.urandom(8).hex(),
            'phi_calculation': 'phi_calc_layer_v1_' + os.urandom(8).hex(),
            'cache': 'cache_layer_v1_' + os.urandom(8).hex(),
            'output': 'output_layer_v1_' + os.urandom(8).hex()
        }
    
    def start_monitoring(self):
        """Inicia monitoreo continuo en background thread"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="SecurityMonitor"
        )
        self.monitor_thread.start()
        
        logger.info("Security monitoring started")
    
    def _monitoring_loop(self):
        """Loop principal de monitoreo"""
        while self.monitoring_active:
            interval = (
                self.adaptive_monitor.monitoring_interval
                if self.adaptive_monitor
                else self.config.monitoring_interval
            )
            
            time.sleep(interval)
            
            try:
                self._perform_security_checks()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _perform_security_checks(self):
        """Realiza todos los checks de seguridad"""
        violations = []
        
        # Check topological integrity
        if self.topo_protection:
            is_valid, tampered_layer = self.topo_protection.verify_integrity()
            if not is_valid:
                violation = {
                    'type': 'TOPOLOGICAL_TAMPERING',
                    'layer': tampered_layer,
                    'severity': 10
                }
                violations.append(violation)
                
                if self.config.mode == SecurityMode.PRODUCTION:
                    logger.critical(
                        f"CRITICAL: Topological tampering in {tampered_layer}"
                    )
        
        # Check file integrity
        if self.file_monitor:
            file_violations = self.file_monitor.verify_all()
            violations.extend(file_violations)
        
        # Record threats
        if self.adaptive_monitor:
            for v in violations:
                self.adaptive_monitor.record_threat(
                    v['type'],
                    v['severity'],
                    details=v
                )
            
            # Adapt parameters (metacognition)
            self.adaptive_monitor.adapt_parameters()
            
            # Check if alert needed
            if self.adaptive_monitor.should_trigger_alert():
                self._trigger_alert(violations)
        
        if not violations:
            logger.debug("Security check: All systems nominal")
    
    def _trigger_alert(self, violations: List[Dict]):
        """Activa alerta de seguridad"""
        logger.critical(
            f"SECURITY ALERT: {len(violations)} violations detected"
        )
        
        for v in violations:
            logger.critical(f"  - {v['type']}: {v.get('path', v.get('layer', 'unknown'))}")
        
        # En producción: Enviar notificación (email, Slack, PagerDuty, etc.)
    
    def verify_before_critical_operation(self) -> bool:
        """
        Verifica integridad antes de operación crítica.
        
        Returns:
            True si puede proceder, False si hay problemas
        """
        if self.topo_protection:
            is_valid, layer = self.topo_protection.verify_integrity()
            if not is_valid:
                logger.error(
                    f"Cannot proceed with critical operation: "
                    f"{layer} layer compromised"
                )
                return False
        
        if self.file_monitor:
            violations = self.file_monitor.verify_all()
            if violations:
                logger.error(
                    f"Cannot proceed: {len(violations)} file violations"
                )
                return False
        
        return True
    
    def get_security_report(self) -> Dict:
        """Genera reporte de seguridad comprehensivo"""
        report = {
            'mode': self.config.mode.name,
            'timestamp': time.time(),
            'monitoring_active': self.monitoring_active
        }
        
        if self.topo_protection:
            is_valid, layer = self.topo_protection.verify_integrity()
            report['topological_protection'] = {
                'enabled': True,
                'status': 'VALID' if is_valid else f'COMPROMISED ({layer})',
                'last_verification': self.topo_protection.last_verification
            }
        
        if self.file_monitor:
            violations = self.file_monitor.verify_all()
            report['file_integrity'] = {
                'enabled': True,
                'monitored_files': len(self.file_monitor.baselines),
                'current_violations': len(violations),
                'violations': violations
            }
        
        if self.adaptive_monitor:
            report['adaptive_monitoring'] = self.adaptive_monitor.get_statistics()
        
        return report
    
    def shutdown(self):
        """Shutdown graceful del módulo"""
        logger.info("Shutting down security module...")
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Security module shutdown complete")

# =============================================================================
# DEMO
# =============================================================================

def demo_security():
    """Demo del módulo de seguridad"""
    print("\n" + "="*70)
    print("ConsciousAI Security Module v2.0 - Demo")
    print("="*70 + "\n")
    
    # Configuration
    config = SecurityConfig(
        mode=SecurityMode.STAGING,
        monitoring_interval=5.0,
        enable_topological_protection=NETWORKX_AVAILABLE,
        enable_file_monitoring=True,
        enable_adaptive_monitoring=True,
        protected_files=[__file__]  # Monitor este mismo archivo
    )
    
    # Initialize
    security = ConsciousAISecurityModule(config)
    security.start_monitoring()
    
    print("✅ Security module initialized")
    print(f"   Mode: {config.mode.name}")
    print(f"   Monitoring interval: {config.monitoring_interval}s")
    print()
    
    # Simulate operation
    print("🔍 Running security checks...")
    time.sleep(6)  # Wait for first check
    
    # Verify before critical operation
    print("\n🔐 Verifying before critical operation...")
    can_proceed = security.verify_before_critical_operation()
    print(f"   Can proceed: {can_proceed}")
    print()
    
    # Generate report
    print("📊 Security Report:")
    report = security.get_security_report()
    print(json.dumps(report, indent=2, default=str))
    print()
    
    # Shutdown
    print("🛑 Shutting down...")
    security.shutdown()
    print("\n✅ Demo complete!\n")

if __name__ == "__main__":
    demo_security()
