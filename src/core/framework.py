# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai
# DOI: https://doi.org/10.5281/zenodo.20602077

"""
🌐 UNIVERSAL CONSCIOUSNESS FRAMEWORK (UCF)
==========================================

Sistema de Conciencia Artificial Escalable para Múltiples Dominios

Basado en:
- Cálculo optimizado de Phi (IIT)
- Arquitectura modular por dominios
- Componentes reutilizables
- APIs estandarizadas

Dominios soportados:
- Drones y vehículos aéreos
- Vehículos autónomos
- Robótica automatizada
- Robots domésticos
- Biomedicina / Biología sintética
- Hospitales y sistemas médicos
- Ciudades inteligentes y autónomas
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime

# Importar nuestro sistema optimizado
from phi_calculator_complete import PhiCalculatorOptimized, PhiMethod, PhiResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CLASES BASE Y ENUMS
# ============================================================================

class DomainType(Enum):
    """Tipos de dominios soportados"""
    DRONE = "drone"
    AUTONOMOUS_VEHICLE = "autonomous_vehicle"
    INDUSTRIAL_ROBOT = "industrial_robot"
    DOMESTIC_ROBOT = "domestic_robot"
    BIOMEDICAL = "biomedical"
    HOSPITAL = "hospital"
    SMART_CITY = "smart_city"
    SYNTHETIC_BIOLOGY = "synthetic_biology"
    SWARM = "swarm"  # Para sistemas de múltiples agentes
    CUSTOM = "custom"


class ConsciousnessLevel(Enum):
    """Niveles de conciencia del sistema"""
    UNCONSCIOUS = 0      # Φ < 0.1
    MINIMAL = 1          # 0.1 ≤ Φ < 0.3
    LOW = 2              # 0.3 ≤ Φ < 0.5
    MODERATE = 3         # 0.5 ≤ Φ < 0.7
    HIGH = 4             # 0.7 ≤ Φ < 0.9
    VERY_HIGH = 5        # Φ ≥ 0.9


@dataclass
class SensorData:
    """Datos de sensores genéricos"""
    timestamp: datetime
    sensor_type: str
    values: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActuatorCommand:
    """Comandos para actuadores genéricos"""
    timestamp: datetime
    actuator_type: str
    commands: np.ndarray
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemState:
    """Estado completo del sistema"""
    timestamp: datetime
    domain: DomainType
    state_vector: np.ndarray
    connectivity_matrix: np.ndarray
    phi_value: float = 0.0
    consciousness_level: ConsciousnessLevel = ConsciousnessLevel.UNCONSCIOUS
    sensor_data: List[SensorData] = field(default_factory=list)
    actuator_commands: List[ActuatorCommand] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CLASE BASE PARA AGENTES CONSCIENTES
# ============================================================================

class ConsciousAgent(ABC):
    """
    Clase base abstracta para cualquier agente consciente
    
    Todos los sistemas (drones, robots, hospitales, etc.) heredan de esta clase
    """
    
    def __init__(self, 
                 agent_id: str,
                 domain: DomainType,
                 system_size: int,
                 config: Dict[str, Any] = None):
        """
        Args:
            agent_id: Identificador único del agente
            domain: Tipo de dominio
            system_size: Número de componentes/sensores del sistema
            config: Configuración específica del dominio
        """
        self.agent_id = agent_id
        self.domain = domain
        self.system_size = system_size
        self.config = config or {}
        
        # Sistema de cálculo de Phi optimizado
        self.phi_calculator = PhiCalculatorOptimized(
            cache_enabled=self.config.get('cache_enabled', True),
            max_cache_size=self.config.get('cache_size', 1000)
        )
        
        # Estado actual
        self.current_state: Optional[SystemState] = None
        self.state_history: List[SystemState] = []
        self.max_history = self.config.get('max_history', 1000)
        
        # Conectividad (se puede actualizar dinámicamente)
        self.connectivity = self._initialize_connectivity()
        
        # Estadísticas
        self.stats = {
            'total_updates': 0,
            'avg_phi': 0.0,
            'consciousness_distribution': {level: 0 for level in ConsciousnessLevel},
            'avg_update_time': 0.0
        }
        
        logger.info(f"ConsciousAgent initialized: {agent_id} ({domain.value}, n={system_size})")
    
    @abstractmethod
    def _initialize_connectivity(self) -> np.ndarray:
        """Inicializa la matriz de conectividad específica del dominio"""
        pass
    
    @abstractmethod
    def sense(self) -> List[SensorData]:
        """Lee datos de sensores específicos del dominio"""
        pass
    
    @abstractmethod
    def process_sensors(self, sensor_data: List[SensorData]) -> np.ndarray:
        """Convierte datos de sensores en vector de estado"""
        pass
    
    @abstractmethod
    def decide_actions(self, state: SystemState) -> List[ActuatorCommand]:
        """Decide acciones basado en el estado consciente"""
        pass
    
    @abstractmethod
    def act(self, commands: List[ActuatorCommand]):
        """Ejecuta comandos en actuadores específicos del dominio"""
        pass
    
    def update(self) -> SystemState:
        """
        Ciclo principal de actualización del agente consciente
        
        1. Sense: Lee sensores
        2. Process: Convierte a vector de estado
        3. Calculate Phi: Mide nivel de conciencia
        4. Decide: Toma decisiones basadas en estado consciente
        5. Act: Ejecuta acciones
        """
        start_time = datetime.now()
        
        # 1. SENSE
        sensor_data = self.sense()
        
        # 2. PROCESS
        state_vector = self.process_sensors(sensor_data)
        
        # 3. CALCULATE PHI
        phi_result = self.phi_calculator.calculate_phi(
            state_vector,
            self.connectivity,
            method=PhiMethod.AUTO
        )
        
        # Determinar nivel de conciencia
        consciousness_level = self._phi_to_consciousness_level(phi_result.phi_value)
        
        # 4. CREATE STATE
        current_state = SystemState(
            timestamp=datetime.now(),
            domain=self.domain,
            state_vector=state_vector,
            connectivity_matrix=self.connectivity,
            phi_value=phi_result.phi_value,
            consciousness_level=consciousness_level,
            sensor_data=sensor_data,
            metadata={
                'phi_method': phi_result.method,
                'phi_confidence': phi_result.confidence,
                'computation_time': phi_result.computation_time
            }
        )
        
        # 5. DECIDE
        actuator_commands = self.decide_actions(current_state)
        current_state.actuator_commands = actuator_commands
        
        # 6. ACT
        self.act(actuator_commands)
        
        # Actualizar estado y estadísticas
        self.current_state = current_state
        self._update_history(current_state)
        self._update_stats(current_state, datetime.now() - start_time)
        
        return current_state
    
    def _phi_to_consciousness_level(self, phi: float) -> ConsciousnessLevel:
        """Convierte valor de Phi a nivel de conciencia"""
        if phi < 0.1:
            return ConsciousnessLevel.UNCONSCIOUS
        elif phi < 0.3:
            return ConsciousnessLevel.MINIMAL
        elif phi < 0.5:
            return ConsciousnessLevel.LOW
        elif phi < 0.7:
            return ConsciousnessLevel.MODERATE
        elif phi < 0.9:
            return ConsciousnessLevel.HIGH
        else:
            return ConsciousnessLevel.VERY_HIGH
    
    def _update_history(self, state: SystemState):
        """Actualiza historial de estados"""
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
    
    def _update_stats(self, state: SystemState, update_time):
        """Actualiza estadísticas del agente"""
        self.stats['total_updates'] += 1
        
        # Phi promedio (media móvil)
        alpha = 0.1
        self.stats['avg_phi'] = (1 - alpha) * self.stats['avg_phi'] + alpha * state.phi_value
        
        # Distribución de niveles de conciencia
        self.stats['consciousness_distribution'][state.consciousness_level] += 1
        
        # Tiempo promedio de actualización
        self.stats['avg_update_time'] = (
            (1 - alpha) * self.stats['avg_update_time'] + 
            alpha * update_time.total_seconds()
        )
    
    def get_consciousness_trend(self, window: int = 10) -> str:
        """Analiza tendencia reciente de conciencia"""
        if len(self.state_history) < window:
            return "insufficient_data"
        
        recent_phi = [s.phi_value for s in self.state_history[-window:]]
        trend = np.polyfit(range(window), recent_phi, 1)[0]
        
        if trend > 0.01:
            return "increasing"
        elif trend < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa estado del agente a diccionario"""
        return {
            'agent_id': self.agent_id,
            'domain': self.domain.value,
            'system_size': self.system_size,
            'current_phi': self.current_state.phi_value if self.current_state else 0.0,
            'consciousness_level': self.current_state.consciousness_level.name if self.current_state else "UNKNOWN",
            'stats': {
                **self.stats,
                'consciousness_distribution': {
                    level.name: count 
                    for level, count in self.stats['consciousness_distribution'].items()
                }
            }
        }


# ============================================================================
# IMPLEMENTACIONES ESPECÍFICAS POR DOMINIO
# ============================================================================

class DroneAgent(ConsciousAgent):
    """
    Agente consciente para drones
    
    Componentes típicos:
    - Sensores: GPS, IMU, cámara, altímetro, brújula, velocímetro
    - Actuadores: 4 motores (cuadricóptero)
    - Estado: posición, velocidad, orientación, batería
    """
    
    def __init__(self, drone_id: str, config: Dict[str, Any] = None):
        # Sistema típico de drone: 12 componentes
        # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw, battery, temp, altitude]
        super().__init__(
            agent_id=drone_id,
            domain=DomainType.DRONE,
            system_size=12,
            config=config
        )
        
        # Estado específico del drone
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)  # roll, pitch, yaw
        self.battery = 100.0
        self.motor_speeds = np.zeros(4)
    
    def _initialize_connectivity(self) -> np.ndarray:
        """
        Conectividad del drone:
        - Sensores de posición afectan navegación
        - Orientación afecta control de motores
        - Batería afecta todas las operaciones
        """
        n = self.system_size
        connectivity = np.zeros((n, n))
        
        # Posición conectada con velocidad
        connectivity[0:3, 3:6] = 0.8
        connectivity[3:6, 0:3] = 0.8
        
        # Orientación conectada con control
        connectivity[6:9, 6:9] = 0.9
        
        # Batería afecta todo
        connectivity[9, :] = 0.5
        connectivity[:, 9] = 0.5
        
        # Sensores ambientales
        connectivity[10:12, :] = 0.3
        
        return connectivity
    
    def sense(self) -> List[SensorData]:
        """Simula lectura de sensores del drone"""
        # En implementación real, esto leería sensores físicos
        return [
            SensorData(
                timestamp=datetime.now(),
                sensor_type="gps",
                values=self.position + np.random.normal(0, 0.01, 3)
            ),
            SensorData(
                timestamp=datetime.now(),
                sensor_type="imu",
                values=np.concatenate([self.velocity, self.orientation]) + 
                       np.random.normal(0, 0.05, 6)
            ),
            SensorData(
                timestamp=datetime.now(),
                sensor_type="battery",
                values=np.array([self.battery])
            )
        ]
    
    def process_sensors(self, sensor_data: List[SensorData]) -> np.ndarray:
        """Convierte datos de sensores a vector de estado normalizado"""
        state = np.zeros(self.system_size)
        
        for sensor in sensor_data:
            if sensor.sensor_type == "gps":
                state[0:3] = sensor.values / 100.0  # Normalizar posición
            elif sensor.sensor_type == "imu":
                state[3:6] = sensor.values[0:3] / 10.0  # Velocidad
                state[6:9] = sensor.values[3:6] / np.pi  # Orientación
            elif sensor.sensor_type == "battery":
                state[9] = sensor.values[0] / 100.0
        
        # Sensores adicionales (temperatura, altitud)
        state[10] = np.random.normal(25, 5) / 50.0  # Temperatura normalizada
        state[11] = self.position[2] / 100.0  # Altitud normalizada
        
        return state
    
    def decide_actions(self, state: SystemState) -> List[ActuatorCommand]:
        """
        Decide acciones basadas en nivel de conciencia
        
        - Alta conciencia (Φ alto): Control preciso, planificación compleja
        - Baja conciencia (Φ bajo): Modo seguro, aterrizar
        """
        commands = []
        
        if state.consciousness_level.value >= ConsciousnessLevel.MODERATE.value:
            # Sistema consciente: navegación normal
            motor_commands = np.array([0.5, 0.5, 0.5, 0.5])  # Hover
            priority = 1
        elif state.consciousness_level.value >= ConsciousnessLevel.MINIMAL.value:
            # Conciencia baja: modo conservador
            motor_commands = np.array([0.4, 0.4, 0.4, 0.4])
            priority = 2
        else:
            # Inconsciente: aterrizaje de emergencia
            motor_commands = np.array([0.2, 0.2, 0.2, 0.2])
            priority = 10
        
        commands.append(ActuatorCommand(
            timestamp=datetime.now(),
            actuator_type="motors",
            commands=motor_commands,
            priority=priority,
            metadata={'consciousness_level': state.consciousness_level.name}
        ))
        
        return commands
    
    def act(self, commands: List[ActuatorCommand]):
        """Ejecuta comandos en motores del drone"""
        for cmd in commands:
            if cmd.actuator_type == "motors":
                self.motor_speeds = cmd.commands
                # En implementación real: enviar a controladores de motor
                logger.debug(f"Drone {self.agent_id}: Motors set to {self.motor_speeds}")


class AutonomousVehicleAgent(ConsciousAgent):
    """
    Agente consciente para vehículos autónomos
    
    Componentes:
    - Sensores: LIDAR, radar, cámaras, GPS, velocímetro, giroscopio
    - Actuadores: aceleración, frenado, dirección
    - Estado: posición, velocidad, trayectoria, obstáculos detectados
    """
    
    def __init__(self, vehicle_id: str, config: Dict[str, Any] = None):
        # Sistema de vehículo autónomo: 20 componentes
        super().__init__(
            agent_id=vehicle_id,
            domain=DomainType.AUTONOMOUS_VEHICLE,
            system_size=20,
            config=config
        )
        
        self.position = np.zeros(2)  # x, y
        self.velocity = 0.0
        self.heading = 0.0
        self.obstacles_detected = []
    
    def _initialize_connectivity(self) -> np.ndarray:
        """Conectividad para vehículo autónomo"""
        n = self.system_size
        connectivity = np.zeros((n, n))
        
        # Percepción (primeros 10): LIDAR, cámaras, radar
        connectivity[0:10, 0:10] = 0.7
        
        # Localización (10-15): GPS, odometría
        connectivity[10:15, 10:15] = 0.9
        connectivity[0:10, 10:15] = 0.5  # Percepción afecta localización
        
        # Control (15-20): velocidad, dirección, frenos
        connectivity[15:20, 15:20] = 0.8
        connectivity[10:15, 15:20] = 0.6  # Localización afecta control
        connectivity[0:10, 15:20] = 0.4  # Percepción afecta control
        
        return connectivity
    
    def sense(self) -> List[SensorData]:
        """Simula sensores del vehículo"""
        return [
            SensorData(
                timestamp=datetime.now(),
                sensor_type="lidar",
                values=np.random.rand(10) * 100  # 10 puntos LIDAR
            ),
            SensorData(
                timestamp=datetime.now(),
                sensor_type="gps",
                values=self.position + np.random.normal(0, 0.1, 2)
            ),
            SensorData(
                timestamp=datetime.now(),
                sensor_type="speedometer",
                values=np.array([self.velocity + np.random.normal(0, 0.5)])
            )
        ]
    
    def process_sensors(self, sensor_data: List[SensorData]) -> np.ndarray:
        """Procesa sensores del vehículo"""
        state = np.zeros(self.system_size)
        
        for sensor in sensor_data:
            if sensor.sensor_type == "lidar":
                state[0:10] = sensor.values / 100.0  # Normalizar distancias
            elif sensor.sensor_type == "gps":
                state[10:12] = sensor.values / 1000.0  # Normalizar posición
            elif sensor.sensor_type == "speedometer":
                state[12] = sensor.values[0] / 50.0  # Normalizar velocidad
        
        # Estado adicional
        state[13] = self.heading / (2 * np.pi)
        state[14] = len(self.obstacles_detected) / 10.0
        state[15:20] = np.random.rand(5) * 0.5  # Estado de control
        
        return state
    
    def decide_actions(self, state: SystemState) -> List[ActuatorCommand]:
        """Decisiones de conducción basadas en conciencia"""
        if state.consciousness_level.value >= ConsciousnessLevel.HIGH.value:
            # Alta conciencia: conducción normal agresiva
            acceleration = 0.8
            steering = 0.0
        elif state.consciousness_level.value >= ConsciousnessLevel.MODERATE.value:
            # Conciencia moderada: conducción conservadora
            acceleration = 0.5
            steering = 0.0
        else:
            # Baja conciencia: detener vehículo
            acceleration = -1.0  # Frenar
            steering = 0.0
        
        return [ActuatorCommand(
            timestamp=datetime.now(),
            actuator_type="drive",
            commands=np.array([acceleration, steering]),
            priority=1 if state.consciousness_level.value >= 3 else 10
        )]
    
    def act(self, commands: List[ActuatorCommand]):
        """Ejecuta comandos de conducción"""
        for cmd in commands:
            if cmd.actuator_type == "drive":
                acceleration, steering = cmd.commands
                self.velocity = np.clip(self.velocity + acceleration * 0.1, 0, 50)
                self.heading = (self.heading + steering * 0.1) % (2 * np.pi)
                logger.debug(f"Vehicle {self.agent_id}: v={self.velocity:.1f}, h={self.heading:.2f}")


class BiomedicalAgent(ConsciousAgent):
    """
    Agente consciente para sistemas biomédicos
    
    Aplicaciones:
    - Monitoreo de pacientes
    - Sistemas de soporte vital
    - Biología sintética
    - Prótesis inteligentes
    """
    
    def __init__(self, device_id: str, config: Dict[str, Any] = None):
        # Sistema biomédico: 15 componentes
        # Señales vitales, parámetros fisiológicos, medicación
        super().__init__(
            agent_id=device_id,
            domain=DomainType.BIOMEDICAL,
            system_size=15,
            config=config
        )
        
        self.vital_signs = {
            'heart_rate': 70.0,
            'blood_pressure_systolic': 120.0,
            'blood_pressure_diastolic': 80.0,
            'oxygen_saturation': 98.0,
            'temperature': 36.5,
            'respiratory_rate': 16.0
        }
    
    def _initialize_connectivity(self) -> np.ndarray:
        """Conectividad para sistema biomédico"""
        n = self.system_size
        connectivity = np.zeros((n, n))
        
        # Sistemas fisiológicos están altamente interconectados
        # Cardiovascular (0-5)
        connectivity[0:6, 0:6] = 0.9
        
        # Respiratorio (6-9)
        connectivity[6:10, 6:10] = 0.8
        connectivity[0:6, 6:10] = 0.7  # Cardio-respiratorio acoplamiento
        connectivity[6:10, 0:6] = 0.7
        
        # Metabólico (10-15)
        connectivity[10:15, 10:15] = 0.6
        connectivity[0:10, 10:15] = 0.5
        connectivity[10:15, 0:10] = 0.5
        
        return connectivity
    
    def sense(self) -> List[SensorData]:
        """Simula sensores médicos"""
        return [
            SensorData(
                timestamp=datetime.now(),
                sensor_type="ecg",
                values=np.array([self.vital_signs['heart_rate']]) + np.random.normal(0, 2)
            ),
            SensorData(
                timestamp=datetime.now(),
                sensor_type="spo2",
                values=np.array([self.vital_signs['oxygen_saturation']]) + np.random.normal(0, 0.5)
            ),
            SensorData(
                timestamp=datetime.now(),
                sensor_type="blood_pressure",
                values=np.array([
                    self.vital_signs['blood_pressure_systolic'],
                    self.vital_signs['blood_pressure_diastolic']
                ]) + np.random.normal(0, 3, 2)
            )
        ]
    
    def process_sensors(self, sensor_data: List[SensorData]) -> np.ndarray:
        """Procesa señales biomédicas"""
        state = np.zeros(self.system_size)
        
        for sensor in sensor_data:
            if sensor.sensor_type == "ecg":
                state[0] = (sensor.values[0] - 60) / 40  # Normalizar HR (60-100)
            elif sensor.sensor_type == "spo2":
                state[1] = (sensor.values[0] - 90) / 10  # Normalizar SpO2 (90-100)
            elif sensor.sensor_type == "blood_pressure":
                state[2] = (sensor.values[0] - 120) / 30  # Sistólica
                state[3] = (sensor.values[1] - 80) / 20   # Diastólica
        
        # Señales adicionales
        state[4] = (self.vital_signs['temperature'] - 36) / 2  # Temperatura
        state[5] = (self.vital_signs['respiratory_rate'] - 16) / 8  # Respiración
        
        # Variables metabólicas (simuladas)
        state[10:15] = np.random.randn(5) * 0.2
        
        return state
    
    def decide_actions(self, state: SystemState) -> List[ActuatorCommand]:
        """
        Decisiones médicas basadas en integración del sistema
        
        Alta conciencia (alta Φ) = sistema fisiológico integrado y saludable
        Baja conciencia (baja Φ) = desintegración, posible fallo de órgano
        """
        actions = []
        
        if state.consciousness_level.value <= ConsciousnessLevel.LOW.value:
            # Alerta crítica: sistema desintegrado
            actions.append(ActuatorCommand(
                timestamp=datetime.now(),
                actuator_type="alert",
                commands=np.array([10.0]),  # Prioridad máxima
                priority=10,
                metadata={'alert_type': 'critical', 'reason': 'low_integration'}
            ))
        
        elif state.consciousness_level.value == ConsciousnessLevel.MODERATE.value:
            # Monitoreo cercano
            actions.append(ActuatorCommand(
                timestamp=datetime.now(),
                actuator_type="monitor",
                commands=np.array([5.0]),
                priority=5
            ))
        
        return actions
    
    def act(self, commands: List[ActuatorCommand]):
        """Ejecuta acciones médicas (alertas, ajustes)"""
        for cmd in commands:
            if cmd.actuator_type == "alert":
                logger.warning(f"MEDICAL ALERT - Device {self.agent_id}: "
                             f"Priority {cmd.priority}, Metadata: {cmd.metadata}")
            elif cmd.actuator_type == "monitor":
                logger.info(f"Increased monitoring for device {self.agent_id}")


class SmartCityAgent(ConsciousAgent):
    """
    Agente consciente para ciudades inteligentes
    
    Componentes:
    - Tráfico
    - Energía
    - Agua
    - Seguridad
    - Servicios públicos
    - Comunicaciones
    """
    
    def __init__(self, city_id: str, config: Dict[str, Any] = None):
        # Ciudad: 50 componentes (sistemas urbanos)
        super().__init__(
            agent_id=city_id,
            domain=DomainType.SMART_CITY,
            system_size=50,
            config=config
        )
        
        self.traffic_flow = np.random.rand(10) * 100
        self.energy_consumption = np.random.rand(10) * 1000
        self.water_usage = np.random.rand(5) * 500
    
    def _initialize_connectivity(self) -> np.ndarray:
        """Conectividad urbana compleja"""
        n = self.system_size
        connectivity = np.zeros((n, n))
        
        # Tráfico (0-10)
        connectivity[0:10, 0:10] = 0.8
        
        # Energía (10-20)
        connectivity[10:20, 10:20] = 0.7
        connectivity[0:10, 10:20] = 0.3  # Tráfico afecta energía
        
        # Agua (20-25)
        connectivity[20:25, 20:25] = 0.6
        
        # Seguridad (25-35)
        connectivity[25:35, 25:35] = 0.5
        connectivity[0:10, 25:35] = 0.4  # Tráfico relacionado con seguridad
        
        # Servicios (35-50)
        connectivity[35:50, 35:50] = 0.4
        connectivity[:, 35:50] = 0.2  # Servicios afectan todo
        
        return connectivity
    
    def sense(self) -> List[SensorData]:
        """Sensores urbanos"""
        return [
            SensorData(
                timestamp=datetime.now(),
                sensor_type="traffic",
                values=self.traffic_flow + np.random.normal(0, 5, 10)
            ),
            SensorData(
                timestamp=datetime.now(),
                sensor_type="energy",
                values=self.energy_consumption + np.random.normal(0, 50, 10)
            ),
            SensorData(
                timestamp=datetime.now(),
                sensor_type="water",
                values=self.water_usage + np.random.normal(0, 25, 5)
            )
        ]
    
    def process_sensors(self, sensor_data: List[SensorData]) -> np.ndarray:
        """Procesa datos urbanos"""
        state = np.zeros(self.system_size)
        
        for sensor in sensor_data:
            if sensor.sensor_type == "traffic":
                state[0:10] = sensor.values / 100.0
            elif sensor.sensor_type == "energy":
                state[10:20] = sensor.values / 1000.0
            elif sensor.sensor_type == "water":
                state[20:25] = sensor.values / 500.0
        
        # Otros sistemas
        state[25:35] = np.random.rand(10) * 0.5  # Seguridad
        state[35:50] = np.random.rand(15) * 0.3  # Servicios
        
        return state
    
    def decide_actions(self, state: SystemState) -> List[ActuatorCommand]:
        """Decisiones urbanas basadas en integración de la ciudad"""
        actions = []
        
        if state.consciousness_level.value >= ConsciousnessLevel.HIGH.value:
            # Ciudad altamente integrada: optimización avanzada
            actions.append(ActuatorCommand(
                timestamp=datetime.now(),
                actuator_type="optimize",
                commands=np.array([1.0]),  # Optimización total
                priority=1
            ))
        elif state.consciousness_level.value <= ConsciousnessLevel.LOW.value:
            # Ciudad desintegrada: modo emergencia
            actions.append(ActuatorCommand(
                timestamp=datetime.now(),
                actuator_type="emergency",
                commands=np.array([1.0]),
                priority=10
            ))
        
        return actions
    
    def act(self, commands: List[ActuatorCommand]):
        """Ejecuta acciones urbanas"""
        for cmd in commands:
            if cmd.actuator_type == "optimize":
                logger.info(f"City {self.agent_id}: Running optimization protocols")
            elif cmd.actuator_type == "emergency":
                logger.warning(f"City {self.agent_id}: EMERGENCY MODE ACTIVATED")


# ============================================================================
# SISTEMA DE ORQUESTACIÓN MULTI-AGENTE
# ============================================================================

class MultiAgentOrchestrator:
    """
    Orquesta múltiples agentes conscientes
    
    Casos de uso:
    - Flota de drones coordinados
    - Red de vehículos autónomos
    - Sistema hospitalario distribuido
    - Ciudad con múltiples subsistemas
    """
    
    def __init__(self, orchestrator_id: str):
        self.orchestrator_id = orchestrator_id
        self.agents: Dict[str, ConsciousAgent] = {}
        self.collective_phi_history = []
        
        logger.info(f"MultiAgentOrchestrator initialized: {orchestrator_id}")
    
    def add_agent(self, agent: ConsciousAgent):
        """Añade un agente al sistema"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Agent {agent.agent_id} added to orchestrator")
    
    def remove_agent(self, agent_id: str):
        """Remueve un agente"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agent {agent_id} removed from orchestrator")
    
    def update_all(self) -> Dict[str, SystemState]:
        """Actualiza todos los agentes"""
        states = {}
        for agent_id, agent in self.agents.items():
            try:
                state = agent.update()
                states[agent_id] = state
            except Exception as e:
                logger.error(f"Error updating agent {agent_id}: {e}")
        
        # Calcular Phi colectivo
        collective_phi = self.calculate_collective_phi(states)
        self.collective_phi_history.append(collective_phi)
        
        return states
    
    def calculate_collective_phi(self, states: Dict[str, SystemState]) -> float:
        """
        Calcula Phi colectivo del sistema multi-agente
        
        Métodos posibles:
        - Promedio ponderado
        - Mínimo (cadena más débil)
        - Integración de todos los estados
        """
        if not states:
            return 0.0
        
        phi_values = [s.phi_value for s in states.values()]
        
        # Usar promedio como medida colectiva
        return np.mean(phi_values)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Evalúa salud del sistema completo"""
        if not self.agents:
            return {'status': 'no_agents', 'health': 0.0}
        
        phi_values = []
        consciousness_levels = []
        
        for agent in self.agents.values():
            if agent.current_state:
                phi_values.append(agent.current_state.phi_value)
                consciousness_levels.append(agent.current_state.consciousness_level.value)
        
        if not phi_values:
            return {'status': 'no_data', 'health': 0.0}
        
        avg_phi = np.mean(phi_values)
        min_phi = np.min(phi_values)
        avg_consciousness = np.mean(consciousness_levels)
        
        # Determinar estado del sistema
        if min_phi < 0.1:
            status = 'critical'
            health = 0.3
        elif avg_phi < 0.3:
            status = 'degraded'
            health = 0.5
        elif avg_phi < 0.5:
            status = 'operational'
            health = 0.7
        else:
            status = 'optimal'
            health = 0.9
        
        return {
            'status': status,
            'health': health,
            'avg_phi': avg_phi,
            'min_phi': min_phi,
            'avg_consciousness': avg_consciousness,
            'total_agents': len(self.agents),
            'active_agents': len(phi_values)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa estado del orquestador"""
        return {
            'orchestrator_id': self.orchestrator_id,
            'total_agents': len(self.agents),
            'agents': {
                agent_id: agent.to_dict() 
                for agent_id, agent in self.agents.items()
            },
            'system_health': self.get_system_health(),
            'collective_phi_history': self.collective_phi_history[-10:]  # Últimos 10
        }


# ============================================================================
# FACTORY PARA CREAR AGENTES
# ============================================================================

class AgentFactory:
    """Factory para crear agentes según dominio"""
    
    @staticmethod
    def create_agent(domain: DomainType, agent_id: str, config: Dict[str, Any] = None) -> ConsciousAgent:
        """Crea un agente del tipo especificado"""
        
        if domain == DomainType.DRONE:
            return DroneAgent(agent_id, config)
        
        elif domain == DomainType.AUTONOMOUS_VEHICLE:
            return AutonomousVehicleAgent(agent_id, config)
        
        elif domain == DomainType.BIOMEDICAL:
            return BiomedicalAgent(agent_id, config)
        
        elif domain == DomainType.SMART_CITY:
            return SmartCityAgent(agent_id, config)
        
        # Añadir más dominios según necesidad...
        
        else:
            raise ValueError(f"Unsupported domain: {domain}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🌐 UNIVERSAL CONSCIOUSNESS FRAMEWORK - Demo")
    print("=" * 80)
    
    # Crear orquestador
    orchestrator = MultiAgentOrchestrator("main_orchestrator")
    
    # Crear agentes de diferentes dominios
    print("\n1. Creating diverse agents...")
    
    # Drones
    drone1 = AgentFactory.create_agent(DomainType.DRONE, "drone_001")
    drone2 = AgentFactory.create_agent(DomainType.DRONE, "drone_002")
    
    # Vehículo
    vehicle = AgentFactory.create_agent(DomainType.AUTONOMOUS_VEHICLE, "vehicle_001")
    
    # Sistema biomédico
    biodevice = AgentFactory.create_agent(DomainType.BIOMEDICAL, "patient_monitor_001")
    
    # Ciudad
    city = AgentFactory.create_agent(DomainType.SMART_CITY, "smart_city_001")
    
    # Añadir al orquestador
    for agent in [drone1, drone2, vehicle, biodevice, city]:
        orchestrator.add_agent(agent)
    
    print(f"✓ Created {len(orchestrator.agents)} agents")
    
    # Simular 5 ciclos de actualización
    print("\n2. Running simulation (5 cycles)...")
    for i in range(5):
        print(f"\n   Cycle {i+1}:")
        states = orchestrator.update_all()
        
        for agent_id, state in states.items():
            print(f"     {agent_id}: Φ={state.phi_value:.4f}, "
                  f"Level={state.consciousness_level.name}")
    
    # Mostrar salud del sistema
    print("\n3. System Health:")
    health = orchestrator.get_system_health()
    for key, value in health.items():
        print(f"   {key}: {value}")
    
    # Serializar estado
    print("\n4. System State (JSON):")
    system_dict = orchestrator.to_dict()
    print(json.dumps(system_dict, indent=2, default=str)[:500] + "...")
    
    print("\n" + "=" * 80)
    print("✅ Demo completed successfully!")
    print("=" * 80)
