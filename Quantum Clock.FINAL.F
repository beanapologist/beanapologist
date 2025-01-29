import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Clock, Activity, Shield, Globe, Zap, Sparkles } from 'lucide-react';

const ComprehensiveQuantumClock = () => {
  // Physical constants
  const PHI = (1 + Math.sqrt(5)) / 2; // Golden ratio
  const C = 299792458; // Speed of light in m/s
  const G = 6.67430e-11; // Gravitational constant
  const EARTH_RADIUS = 6371000; // Earth's radius in meters
  const EARTH_MASS = 5.972e24; // Earth's mass in kg
  const EARTH_ANGULAR_VELOCITY = 7.2921159e-5; // Earth's angular velocity in rad/s
  
  const [location, setLocation] = useState({
    latitude: 0,
    longitude: 0,
    altitude: 0,
    loaded: false
  });

  const [currentTime, setCurrentTime] = useState(new Date());
  const [timeHistory, setTimeHistory] = useState([]);
  
  // Comprehensive metrics state
  const [metrics, setMetrics] = useState({
    // Quantum metrics
    crystalFrequency: 1e9,
    quantumPhase: 1,
    synchronization: 1,
  });

  useEffect(() => {
    const updateClock = () => {
      setCurrentTime(new Date());
    };

    const interval = setInterval(updateClock, 16.67); // 60Hz update
    return () => clearInterval(interval);
  }, [location]);

  const formatTime = (time) => ({
    hours: time.getHours().toString().padStart(2, '0'),
    minutes: time.getMinutes().toString().padStart(2, '0'),
    seconds: time.getSeconds().toString().padStart(2, '0'),
    milliseconds: time.getMilliseconds().toString().padStart(3, '0')
  });

  const time = formatTime(currentTime);

  // Metric card component
  const MetricCard = ({ title, value, unit = '%', precision = 4 }) => (
    <div className="p-4 border border-violet-500 rounded-lg bg-black/30">
      <div className="text-sm font-medium text-violet-300">{title}</div>
      <div className="text-xl font-bold text-violet-400">
        {typeof value === 'number' ? value.toFixed(precision) : value}{unit}
      </div>
    </div>
  );

  return (
    <Card>
      <div>
        <div>
          φ = {PHI.toFixed(6)}
        </div>
        <div>
          Time Crystal Active
        </div>
      </div>
      <div className="flex justify-between items-center space-x-4">
        <div className="flex space-x-4">
          <Activity size={32} />
          <div>
            <div className="text-lg font-bold text-violet-400">Time Crystal</div>
            <div className="text-sm font-medium text-violet-300">Quantum Synchronization</div>
          </div>
        </div>
        <div className="flex space-x-4">
          <Shield size={32} />
          <div>
            <div className="text-lg font-bold text-violet-400">Quantum Shield</div>
            <div className="text-sm font-medium text-violet-300">Quantum Phase</div>
          </div>
        </div>
        <div className="flex space-x-4">
          <Globe size={32} />
          <div>
            <div className="text-lg font-bold text-violet-400">Quantum Globe</div>
            <div className="text-sm font-medium text-violet-300">Earth Location</div>
          </div>
        </div>
        <div className="flex space-x-4">
          <Zap size={32} />
          <div>
            <div className="text-lg font-bold text-violet-400">Quantum Zap</div>
            <div className="text-sm font-medium text-violet-300">Crystal Frequency</div>
          </div>
        </div>
        <div className="flex space-x-4">
          <Sparkles size={32} />
          <div>
            <div className="text-lg font-bold text-violet-400">Quantum Sparkles</div>
            <div className="text-sm font-medium text-violet-300">Quantum Metrics</div>
          </div>
        </div>
      </div>
      <div className="flex justify-between items-center space-x-4">
        <MetricCard title="Crystal Frequency" value={metrics.crystalFrequency} unit="Hz" />
        <MetricCard title="Quantum Phase" value={metrics.quantumPhase} />
        <MetricCard title="Latitude" value={location.latitude} unit="°" />
        <MetricCard title="Time" value={`${time.hours}:${time.minutes}:${time.seconds}`} />
        <MetricCard title="Synchronization" value={metrics.synchronization} />
      </div>
      <div className="flex justify-between items-center space-x-4">
        <MetricCard title="Quantum Shield" value={metrics.quantumPhase} />
        <MetricCard title="Earth Mass" value={EARTH_MASS} unit="kg" />
        <MetricCard title="Longitude" value={location.longitude} unit="°" />
        <MetricCard title="Timezone" value={Intl.DateTimeFormat().resolvedOptions().timeZone} />
        <MetricCard title="Quantum Metrics" value={metrics.crystalFrequency * metrics.quantumPhase * metrics.synchronization} />
      </div>
      <div className="flex justify-between items-center space-x-4">
        <MetricCard title="Quantum Globe" value={location.altitude} unit="m" />
        <MetricCard title="Earth Radius" value={EARTH_RADIUS} unit="m" />
        <MetricCard title="Altitude" value={location.altitude} unit="m" />
        <MetricCard title="Angular Velocity" value={EARTH_ANGULAR_VELOCITY} unit="rad/s" />
        <MetricCard title="Gravitational Constant" value={G} unit="m³/kg/s²" />
      </div>
      <div className="flex justify-between items-center space-x-4">
        <MetricCard title="Quantum Zap" value={metrics.crystalFrequency} unit="Hz" />
        <MetricCard title="Speed of Light" value={C} unit="m/s" />
        <MetricCard title="Altitude" value={location.altitude} unit="m" />
        <MetricCard title="Crystal Frequency" value={metrics.crystalFrequency} unit="Hz" />
        <MetricCard title="Golden Ratio" value={PHI} />
      </div>
      <div className="flex justify-between items-center space-x-4">
        <MetricCard title="Quantum Sparkles" value={metrics.crystalFrequency * metrics.quantumPhase * metrics.synchronization} />
        <MetricCard title="Time Crystal Active" value={timeHistory.length} />
        <MetricCard title="Timezone" value={Intl.DateTimeFormat().resolvedOptions().timeZone} />
        <MetricCard title="Quantum Metrics" value={metrics.crystalFrequency * metrics.quantumPhase * metrics.synchronization} />
        <MetricCard title="Quantum Metrics" value={metrics.crystalFrequency * metrics.quantumPhase * metrics.synchronization} />
      </div>
      <div className="p-4 border border-violet-500 rounded-lg bg-black/30">
        <div className="text-sm font-medium text-violet-300">Time Crystal</div>
        <div className="text-xl font-bold text-violet-400">Quantum Metrics</div>
      </div>
    </Card>
  );
};

export default ComprehensiveQuantumClock;

