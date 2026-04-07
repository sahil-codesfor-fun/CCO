import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
    LineChart, Line, AreaChart, Area, BarChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    Legend, ReferenceLine, ComposedChart,
} from 'recharts';
import { runSimulation, compareAllAgents, TRAFFIC_PROFILES, AGENTS } from './engine';

// ═══════════════════════════════════════════════════════════════
// Custom Tooltip
// ═══════════════════════════════════════════════════════════════

const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div style={{
            background: 'rgba(10,12,26,0.95)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '12px',
            padding: '12px 16px',
            backdropFilter: 'blur(20px)',
            fontFamily: "'Inter', sans-serif",
        }}>
            <p style={{ color: '#8892b0', fontSize: '0.75rem', marginBottom: '8px', fontWeight: 600 }}>
                Step {label}
            </p>
            {payload.map((entry, i) => (
                <p key={i} style={{ color: entry.color, fontSize: '0.8rem', fontWeight: 500, lineHeight: 1.8 }}>
                    {entry.name}: <strong style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                        {typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}
                    </strong>
                </p>
            ))}
        </div>
    );
};

// ═══════════════════════════════════════════════════════════════
// Metric Card Component
// ═══════════════════════════════════════════════════════════════

const MetricCard = ({ value, label, gradient = 'gradient-cyan', icon, suffix = '' }) => (
    <div className="metric-card">
        <div style={{ fontSize: '1.5rem', marginBottom: '8px' }}>{icon}</div>
        <div className={`metric-value ${gradient}`}>
            {typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value}{suffix}
        </div>
        <div className="metric-label">{label}</div>
    </div>
);

// ═══════════════════════════════════════════════════════════════
// Score Badge
// ═══════════════════════════════════════════════════════════════

const ScoreBadge = ({ score }) => {
    const cls = score >= 0.7 ? 'score-high' : score >= 0.4 ? 'score-mid' : 'score-low';
    return (
        <span className={`score-badge ${cls}`}>
            {(score * 100).toFixed(1)}%
        </span>
    );
};

// ═══════════════════════════════════════════════════════════════
// Background Elements (AI/Tech Theme)
// ═══════════════════════════════════════════════════════════════

const BackgroundElements = () => {
    const robots = ['🤖', '🦾', '🦿', '🛰️', '📟', '💻'];

    return (
        <div className="floating-elements">
            {/* Generate 8 floating robots with random properties */}
            {Array.from({ length: 8 }).map((_, i) => (
                <div
                    key={`robot-${i}`}
                    className="floating-robot"
                    style={{
                        top: `${Math.random() * 100}%`,
                        left: `${Math.random() * 100}%`,
                        '--duration': `${15 + Math.random() * 10}s`,
                        '--x': `${-50 + Math.random() * 100}px`,
                        '--y': `${-50 + Math.random() * 100}px`,
                        '--rot': `${-20 + Math.random() * 40}deg`,
                        fontSize: `${1 + Math.random() * 2}rem`,
                        animationDelay: `${-Math.random() * 10}s`
                    }}
                >
                    {robots[i % robots.length]}
                </div>
            ))}

            {/* Generate 20 floating tech particles */}
            {Array.from({ length: 20 }).map((_, i) => (
                <div
                    key={`tech-${i}`}
                    className="floating-tech"
                    style={{
                        left: `${Math.random() * 100}%`,
                        '--duration': `${5 + Math.random() * 15}s`,
                        animationDelay: `${-Math.random() * 10}s`
                    }}
                />
            ))}
        </div>
    );
};

// ═══════════════════════════════════════════════════════════════
// Navigation
// ═══════════════════════════════════════════════════════════════

const Navbar = ({ activeSection, onNavigate }) => {
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handler = () => setScrolled(window.scrollY > 30);
        window.addEventListener('scroll', handler);
        return () => window.removeEventListener('scroll', handler);
    }, []);

    return (
        <nav className="navbar" style={scrolled ? { boxShadow: '0 4px 30px rgba(0,0,0,0.4)' } : {}}>
            <div className="navbar-inner">
                <div className="navbar-brand">
                    <div className="navbar-logo">☁️</div>
                    <span className="navbar-title">Cloud Cost Optimizer</span>
                    <span className="navbar-tag">AI</span>
                </div>
                <ul className="navbar-links">
                    {['home', 'simulator', 'compare', 'architecture', 'portal'].map(s => (
                        <li key={s}>
                            <a
                                href={`#${s}`}
                                className={activeSection === s ? 'active' : ''}
                                onClick={(e) => { e.preventDefault(); onNavigate(s); }}
                            >
                                {s.charAt(0).toUpperCase() + s.slice(1)}
                            </a>
                        </li>
                    ))}
                </ul>
            </div>
        </nav>
    );
};

// ═══════════════════════════════════════════════════════════════
// Hero Section
// ═══════════════════════════════════════════════════════════════

const Hero = ({ onNavigate }) => (
    <section className="hero" id="home">
        <div className="hero-scanner" />
        <div className="hero-bg-grid" />

        <div className="hero-content">
            <div className="hero-badge">
                <span className="hero-badge-dot" />
                OpenEnv Compatible · Reinforcement Learning
            </div>

            <h1 className="hero-title">
                Intelligent <span className="hero-title-gradient">Cloud Auto-Scaling</span>
                <br />Powered by AI
            </h1>

            <p className="hero-description">
                A production-grade RL environment simulating real-world cloud infrastructure.
                An AI agent manages a fleet of servers, making real-time scaling decisions to
                balance <strong>SLA compliance</strong> against <strong>operational cost</strong> —
                mirroring systems like Kubernetes HPA and AWS Auto Scaling.
            </p>

            <div className="hero-actions">
                <button className="btn btn-primary" onClick={() => onNavigate('simulator')}>
                    🚀 Launch Simulator
                </button>
                <button className="btn btn-secondary" onClick={() => onNavigate('compare')}>
                    📊 Compare Agents
                </button>
                <button className="btn btn-secondary" onClick={() => onNavigate('architecture')}>
                    🏗️ Architecture
                </button>
            </div>

            <div className="hero-stats">
                <div className="hero-stat">
                    <div className="hero-stat-value">5</div>
                    <div className="hero-stat-label">Agent Strategies</div>
                </div>
                <div className="hero-stat">
                    <div className="hero-stat-value">3</div>
                    <div className="hero-stat-label">Difficulty Levels</div>
                </div>
                <div className="hero-stat">
                    <div className="hero-stat-value">11D</div>
                    <div className="hero-stat-label">Observation Space</div>
                </div>
                <div className="hero-stat">
                    <div className="hero-stat-value">0–1</div>
                    <div className="hero-stat-label">Graded Score</div>
                </div>
            </div>
        </div>
    </section>
);

// ═══════════════════════════════════════════════════════════════
// Simulator Section
// ═══════════════════════════════════════════════════════════════

const Simulator = () => {
    const [profile, setProfile] = useState('steady');
    const [agent, setAgent] = useState('threshold');
    const [seed, setSeed] = useState(42);
    const [result, setResult] = useState(null);
    const [isRunning, setIsRunning] = useState(false);
    const [animStep, setAnimStep] = useState(0);
    const [isAnimating, setIsAnimating] = useState(false);

    const handleRun = useCallback(() => {
        setIsRunning(true);
        setIsAnimating(false);
        setAnimStep(0);
        setTimeout(() => {
            const res = runSimulation(profile, agent, seed);
            setResult(res);
            setIsRunning(false);
            // Start animation
            setIsAnimating(true);
        }, 300);
    }, [profile, agent, seed]);

    // Animate chart data
    useEffect(() => {
        if (!isAnimating || !result) return;
        if (animStep >= result.history.traffic.length) {
            setIsAnimating(false);
            return;
        }
        const timer = setTimeout(() => setAnimStep(prev => Math.min(prev + 4, result.history.traffic.length)), 16);
        return () => clearTimeout(timer);
    }, [isAnimating, animStep, result]);

    // Auto-run on mount
    useEffect(() => {
        handleRun();
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const visibleData = useMemo(() => {
        if (!result) return [];
        const len = isAnimating ? animStep : result.history.traffic.length;
        const data = [];
        for (let i = 0; i < len; i++) {
            data.push({
                step: i,
                traffic: result.history.traffic[i],
                served: result.history.served[i],
                servers: result.history.servers[i],
                cpuLoad: result.history.cpuLoad[i] * 100,
                latency: result.history.latency[i],
                dropped: result.history.dropped[i],
                cumCost: result.history.cumCost[i],
            });
        }
        return data;
    }, [result, animStep, isAnimating]);

    return (
        <section className="section section-technical" id="simulator">
            <div className="app-container">
                <h2 className="section-title">🚀 Live Simulator</h2>
                <p className="section-subtitle">
                    Configure a traffic scenario and agent strategy, then watch the auto-scaler in action.
                </p>

                {/* Controls */}
                <div className="sim-panel">
                    <div className="sim-panel-header">
                        <div className="sim-panel-title">
                            ⚙️ Simulation Configuration
                        </div>
                        <div className="sim-controls">
                            <div className="control-group">
                                <label className="control-label">Traffic Scenario</label>
                                <select className="control-select" value={profile} onChange={e => setProfile(e.target.value)}>
                                    <option value="steady">🟢 Steady (Easy)</option>
                                    <option value="spike">🟡 Spike (Medium)</option>
                                    <option value="chaos">🔴 Chaos (Hard)</option>
                                </select>
                            </div>
                            <div className="control-group">
                                <label className="control-label">Agent Strategy</label>
                                <select className="control-select" value={agent} onChange={e => setAgent(e.target.value)}>
                                    {Object.entries(AGENTS).map(([key, a]) => (
                                        <option key={key} value={key}>{a.name}</option>
                                    ))}
                                </select>
                            </div>
                            <div className="control-group">
                                <label className="control-label">Seed</label>
                                <input
                                    type="number"
                                    className="control-select"
                                    style={{ width: '80px' }}
                                    value={seed}
                                    onChange={e => setSeed(Number(e.target.value))}
                                    min={1} max={9999}
                                />
                            </div>
                            <div className="control-group" style={{ justifyContent: 'flex-end' }}>
                                <button className="btn btn-primary" onClick={handleRun} disabled={isRunning}
                                    style={{ padding: '10px 28px', fontSize: '0.85rem' }}>
                                    {isRunning ? '⏳ Running...' : '▶ Run Simulation'}
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Scenario Info */}
                    <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                        <span className={`tag tag-${TRAFFIC_PROFILES[profile].difficulty}`}>
                            {TRAFFIC_PROFILES[profile].difficulty}
                        </span>
                        <span className="tag tag-rl">{TRAFFIC_PROFILES[profile].maxSteps} steps</span>
                        <span className="tag tag-rl">{TRAFFIC_PROFILES[profile].spikes.length} spike events</span>
                    </div>
                </div>

                {/* Results */}
                {result && (
                    <>
                        {/* Score & Metrics */}
                        <div style={{ marginBottom: '32px' }}>
                            <div style={{
                                display: 'flex', alignItems: 'center', gap: '20px',
                                marginBottom: '24px', flexWrap: 'wrap'
                            }}>
                                <h3 style={{ fontSize: '1.3rem', fontWeight: 700 }}>
                                    📊 Results: {result.agent} on {result.profile}
                                </h3>
                                <ScoreBadge score={result.scores.total} />
                            </div>

                            <div className="grid-4" style={{ marginBottom: '24px' }}>
                                <MetricCard value={result.scores.total.toFixed(3)} label="Overall Score" gradient="gradient-cyan" icon="🏆" />
                                <MetricCard value={`$${result.metrics.totalCost}`} label="Total Cost" gradient="gradient-green" icon="💰" />
                                <MetricCard value={`${result.metrics.dropRate}%`} label="Drop Rate" gradient="gradient-red" icon="❌" />
                                <MetricCard value={`${result.metrics.avgLatency}ms`} label="Avg Latency" gradient="gradient-orange" icon="⏱️" />
                            </div>

                            <div className="grid-4">
                                <MetricCard value={result.scores.sla.toFixed(3)} label="SLA Score (40%)" gradient="gradient-cyan" icon="🛡️" />
                                <MetricCard value={result.scores.cost.toFixed(3)} label="Cost Score (30%)" gradient="gradient-green" icon="📉" />
                                <MetricCard value={result.scores.latency.toFixed(3)} label="Latency Score (20%)" gradient="gradient-orange" icon="⚡" />
                                <MetricCard value={result.scores.stability.toFixed(3)} label="Stability Score (10%)" gradient="gradient-purple" icon="📐" />
                            </div>
                        </div>

                        {/* Charts */}
                        <div className="grid-2" style={{ marginBottom: '24px' }}>
                            {/* Traffic Chart */}
                            <div className="chart-container">
                                <div className="chart-title"><span className="chart-title-icon">📈</span> Traffic vs Served Requests</div>
                                <ResponsiveContainer width="100%" height={260}>
                                    <ComposedChart data={visibleData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                        <XAxis dataKey="step" stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <YAxis stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Area type="monotone" dataKey="traffic" fill="rgba(0,210,255,0.1)" stroke="#00d2ff" strokeWidth={2} name="Incoming" />
                                        <Line type="monotone" dataKey="served" stroke="#3a7bd5" strokeWidth={2} dot={false} name="Served" />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Servers Chart */}
                            <div className="chart-container">
                                <div className="chart-title"><span className="chart-title-icon">🖥️</span> Active Servers</div>
                                <ResponsiveContainer width="100%" height={260}>
                                    <AreaChart data={visibleData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                        <XAxis dataKey="step" stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <YAxis stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Area type="stepAfter" dataKey="servers" fill="rgba(118,75,162,0.2)" stroke="#764ba2" strokeWidth={2} name="Servers" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>

                            {/* CPU Load */}
                            <div className="chart-container">
                                <div className="chart-title"><span className="chart-title-icon">📊</span> CPU Load (%)</div>
                                <ResponsiveContainer width="100%" height={260}>
                                    <AreaChart data={visibleData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                        <XAxis dataKey="step" stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <YAxis stroke="#5a6180" tick={{ fontSize: 11 }} domain={[0, 100]} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <ReferenceLine y={75} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'Scale Up', fill: '#ef4444', fontSize: 10 }} />
                                        <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="5 5" label={{ value: 'Scale Down', fill: '#22c55e', fontSize: 10 }} />
                                        <Area type="monotone" dataKey="cpuLoad" fill="rgba(247,151,30,0.15)" stroke="#f7971e" strokeWidth={2} name="CPU %" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Latency */}
                            <div className="chart-container">
                                <div className="chart-title"><span className="chart-title-icon">⏱️</span> Latency (ms)</div>
                                <ResponsiveContainer width="100%" height={260}>
                                    <ComposedChart data={visibleData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                        <XAxis dataKey="step" stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <YAxis stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <ReferenceLine y={200} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'SLA: 200ms', fill: '#ef4444', fontSize: 10 }} />
                                        <Area type="monotone" dataKey="latency" fill="rgba(235,51,73,0.1)" stroke="#eb3349" strokeWidth={2} name="Latency (ms)" />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Dropped Requests */}
                            <div className="chart-container">
                                <div className="chart-title"><span className="chart-title-icon">❌</span> Dropped Requests</div>
                                <ResponsiveContainer width="100%" height={260}>
                                    <BarChart data={visibleData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                        <XAxis dataKey="step" stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <YAxis stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Bar dataKey="dropped" fill="rgba(239,68,68,0.6)" name="Dropped" radius={[2, 2, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Cumulative Cost */}
                            <div className="chart-container">
                                <div className="chart-title"><span className="chart-title-icon">💰</span> Cumulative Cost ($)</div>
                                <ResponsiveContainer width="100%" height={260}>
                                    <AreaChart data={visibleData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                        <XAxis dataKey="step" stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <YAxis stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Area type="monotone" dataKey="cumCost" fill="rgba(150,201,61,0.15)" stroke="#96c93d" strokeWidth={2} name="Cost ($)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Detailed Stats */}
                        <div className="sim-panel" style={{ marginTop: '32px' }}>
                            <div className="sim-panel-title" style={{ marginBottom: '20px' }}>
                                📋 Episode Statistics
                            </div>
                            <div className="grid-4">
                                <div><span style={{ color: '#8892b0', fontSize: '0.75rem' }}>Total Requests</span><br /><strong style={{ fontFamily: "'JetBrains Mono'" }}>{result.metrics.totalRequests.toLocaleString()}</strong></div>
                                <div><span style={{ color: '#8892b0', fontSize: '0.75rem' }}>Served</span><br /><strong style={{ fontFamily: "'JetBrains Mono'", color: '#22c55e' }}>{result.metrics.totalServed.toLocaleString()}</strong></div>
                                <div><span style={{ color: '#8892b0', fontSize: '0.75rem' }}>Dropped</span><br /><strong style={{ fontFamily: "'JetBrains Mono'", color: '#ef4444' }}>{result.metrics.totalDropped.toLocaleString()}</strong></div>
                                <div><span style={{ color: '#8892b0', fontSize: '0.75rem' }}>SLA Compliance</span><br /><strong style={{ fontFamily: "'JetBrains Mono'", color: '#00d2ff' }}>{result.metrics.slaCompliance}%</strong></div>
                                <div><span style={{ color: '#8892b0', fontSize: '0.75rem' }}>Max Latency</span><br /><strong style={{ fontFamily: "'JetBrains Mono'" }}>{result.metrics.maxLatency} ms</strong></div>
                                <div><span style={{ color: '#8892b0', fontSize: '0.75rem' }}>Avg CPU Load</span><br /><strong style={{ fontFamily: "'JetBrains Mono'" }}>{result.metrics.avgCpu}%</strong></div>
                                <div><span style={{ color: '#8892b0', fontSize: '0.75rem' }}>Peak Servers</span><br /><strong style={{ fontFamily: "'JetBrains Mono'" }}>{result.metrics.peakServers}</strong></div>
                                <div><span style={{ color: '#8892b0', fontSize: '0.75rem' }}>Total Reward</span><br /><strong style={{ fontFamily: "'JetBrains Mono'", color: '#a5b4fc' }}>{result.metrics.totalReward}</strong></div>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </section>
    );
};

// ═══════════════════════════════════════════════════════════════
// Agent Comparison Section
// ═══════════════════════════════════════════════════════════════

const CompareSection = () => {
    const [profile, setProfile] = useState('spike');
    const [results, setResults] = useState(null);
    const [isRunning, setIsRunning] = useState(false);

    const handleCompare = useCallback(() => {
        setIsRunning(true);
        setTimeout(() => {
            const res = compareAllAgents(profile, 42);
            setResults(res);
            setIsRunning(false);
        }, 200);
    }, [profile]);

    const radarData = useMemo(() => {
        if (!results) return [];
        return [
            { metric: 'SLA', ...Object.fromEntries(results.map(r => [r.key, r.scores.sla])) },
            { metric: 'Cost', ...Object.fromEntries(results.map(r => [r.key, r.scores.cost])) },
            { metric: 'Latency', ...Object.fromEntries(results.map(r => [r.key, r.scores.latency])) },
            { metric: 'Stability', ...Object.fromEntries(results.map(r => [r.key, r.scores.stability])) },
        ];
    }, [results]);

    // Auto-run comparison on mount
    useEffect(() => {
        handleCompare();
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const radarColors = ['#00d2ff', '#764ba2', '#f7971e', '#96c93d', '#eb3349'];

    return (
        <section className="section section-technical" id="compare">
            <div className="app-container">
                <h2 className="section-title">📊 Agent Comparison</h2>
                <p className="section-subtitle">
                    Pit all agent strategies against each other on the same traffic scenario.
                </p>

                <div className="sim-panel">
                    <div className="sim-panel-header">
                        <div className="sim-panel-title">🏆 Head-to-Head Battle</div>
                        <div className="sim-controls">
                            <div className="control-group">
                                <label className="control-label">Traffic Scenario</label>
                                <select className="control-select" value={profile} onChange={e => setProfile(e.target.value)}>
                                    <option value="steady">🟢 Steady (Easy)</option>
                                    <option value="spike">🟡 Spike (Medium)</option>
                                    <option value="chaos">🔴 Chaos (Hard)</option>
                                </select>
                            </div>
                            <div className="control-group" style={{ justifyContent: 'flex-end' }}>
                                <button className="btn btn-primary" onClick={handleCompare} disabled={isRunning}
                                    style={{ padding: '10px 28px', fontSize: '0.85rem' }}>
                                    {isRunning ? '⏳ Comparing...' : '⚔️ Compare All Agents'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {results && (
                    <>
                        {/* Leaderboard Table */}
                        <div className="sim-panel" style={{ overflowX: 'auto' }}>
                            <div className="sim-panel-title" style={{ marginBottom: '20px' }}>
                                🏅 Leaderboard
                            </div>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Agent</th>
                                        <th>Score</th>
                                        <th>SLA</th>
                                        <th>Cost</th>
                                        <th>Latency</th>
                                        <th>Stability</th>
                                        <th>Cost ($)</th>
                                        <th>Drop Rate</th>
                                        <th>Avg Latency</th>
                                        <th>Peak Servers</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {results.map((r, i) => (
                                        <tr key={r.key}>
                                            <td style={{ fontWeight: 700, color: i === 0 ? '#ffd700' : i === 1 ? '#c0c0c0' : i === 2 ? '#cd7f32' : '#8892b0' }}>
                                                {i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i + 1}`}
                                            </td>
                                            <td style={{ fontWeight: 600 }}>{r.agent}</td>
                                            <td><ScoreBadge score={r.scores.total} /></td>
                                            <td className="mono">{r.scores.sla.toFixed(3)}</td>
                                            <td className="mono">{r.scores.cost.toFixed(3)}</td>
                                            <td className="mono">{r.scores.latency.toFixed(3)}</td>
                                            <td className="mono">{r.scores.stability.toFixed(3)}</td>
                                            <td className="mono">${r.metrics.totalCost}</td>
                                            <td className="mono" style={{ color: r.metrics.dropRate > 1 ? '#ef4444' : '#22c55e' }}>{r.metrics.dropRate}%</td>
                                            <td className="mono">{r.metrics.avgLatency}ms</td>
                                            <td className="mono">{r.metrics.peakServers}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* Radar Chart */}
                        <div className="grid-2" style={{ marginTop: '24px' }}>
                            <div className="chart-container">
                                <div className="chart-title"><span className="chart-title-icon">🎯</span> Performance Radar</div>
                                <ResponsiveContainer width="100%" height={400}>
                                    <RadarChart data={radarData}>
                                        <PolarGrid stroke="rgba(255,255,255,0.08)" />
                                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#8892b0', fontSize: 12 }} />
                                        <PolarRadiusAxis tick={{ fill: '#5a6180', fontSize: 10 }} domain={[0, 1]} />
                                        {results.map((r, i) => (
                                            <Radar
                                                key={r.key}
                                                name={r.agent}
                                                dataKey={r.key}
                                                stroke={radarColors[i]}
                                                fill={radarColors[i]}
                                                fillOpacity={0.1}
                                                strokeWidth={2}
                                            />
                                        ))}
                                        <Legend wrapperStyle={{ fontSize: '0.75rem', color: '#8892b0' }} />
                                    </RadarChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Score Bar Comparison */}
                            <div className="chart-container">
                                <div className="chart-title"><span className="chart-title-icon">📊</span> Overall Scores</div>
                                <ResponsiveContainer width="100%" height={400}>
                                    <BarChart data={results.map(r => ({ name: r.key, score: r.scores.total * 100 }))} layout="vertical">
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                        <XAxis type="number" domain={[0, 100]} stroke="#5a6180" tick={{ fontSize: 11 }} />
                                        <YAxis type="category" dataKey="name" stroke="#5a6180" tick={{ fontSize: 11 }} width={90} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Bar dataKey="score" name="Score %" radius={[0, 6, 6, 0]}>
                                            {results.map((r, i) => (
                                                <rect key={r.key} fill={radarColors[i % radarColors.length]} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </section>
    );
};

// ═══════════════════════════════════════════════════════════════
// Architecture Section
// ═══════════════════════════════════════════════════════════════

const Architecture = () => (
    <section className="section section-technical" id="architecture">
        <div className="app-container">
            <h2 className="section-title">🏗️ System Architecture</h2>
            <p className="section-subtitle">
                How the Cloud Cost Optimizer simulation works — from traffic generation to reward computation.
            </p>

            {/* Pipeline */}
            <div className="arch-flow">
                {[
                    { icon: '📡', label: 'Traffic Generator', sub: 'Workload Patterns' },
                    { icon: '→' },
                    { icon: '🌐', label: 'Environment', sub: 'Server Fleet Sim' },
                    { icon: '→' },
                    { icon: '🤖', label: 'RL Agent', sub: 'Scaling Decisions' },
                    { icon: '→' },
                    { icon: '📊', label: 'Reward Engine', sub: 'Multi-objective Scorer' },
                    { icon: '→' },
                    { icon: '🏆', label: 'Grader', sub: '0.0 – 1.0 Score' },
                ].map((node, i) => (
                    node.label ? (
                        <div className="arch-node" key={i}>
                            <div style={{ fontSize: '1.5rem', marginBottom: '6px' }}>{node.icon}</div>
                            <div style={{ fontWeight: 700 }}>{node.label}</div>
                            <div className="arch-node-label">{node.sub}</div>
                        </div>
                    ) : (
                        <span className="arch-arrow" key={i}>{node.icon}</span>
                    )
                ))}
            </div>

            {/* Feature Cards */}
            <div className="grid-3" style={{ marginTop: '40px' }}>
                <div className="card fade-in delay-1">
                    <div className="card-icon card-icon-cyan">📡</div>
                    <div className="card-title">Realistic Traffic Modeling</div>
                    <div className="card-text">
                        5-component signal: sinusoidal seasonality, scheduled flash-sale spikes
                        (bell-curve shaped), random micro-bursts, Gaussian noise, and organic growth trend.
                    </div>
                </div>

                <div className="card fade-in delay-2">
                    <div className="card-icon card-icon-purple">🖥️</div>
                    <div className="card-title">Server Fleet Simulation</div>
                    <div className="card-text">
                        Warm-up delay (3 steps), exponential latency near saturation,
                        hard request dropping beyond capacity, and per-server cost model.
                    </div>
                </div>

                <div className="card fade-in delay-3">
                    <div className="card-icon card-icon-orange">⚡</div>
                    <div className="card-title">Latency Model</div>
                    <div className="card-text">
                        latency = base × e^(cpu_load × k). At low utilization it's fast; near saturation,
                        latency explodes — forcing the agent to scale proactively.
                    </div>
                </div>

                <div className="card fade-in delay-1">
                    <div className="card-icon card-icon-red">🎯</div>
                    <div className="card-title">Multi-Objective Reward</div>
                    <div className="card-text">
                        Dense signal balancing 5 competing objectives: served request reward, server cost,
                        dropped request penalty (100x), latency penalty, and efficiency bonus.
                    </div>
                </div>

                <div className="card fade-in delay-2">
                    <div className="card-icon card-icon-green">📐</div>
                    <div className="card-title">Deterministic Grading</div>
                    <div className="card-text">
                        4-component scoring (0.0–1.0): SLA compliance (40%), cost efficiency (30%),
                        latency quality (20%), and scaling stability (10%). Fully reproducible.
                    </div>
                </div>

                <div className="card fade-in delay-3">
                    <div className="card-icon card-icon-blue">🔌</div>
                    <div className="card-title">OpenEnv Compatible</div>
                    <div className="card-text">
                        Full interface: typed Pydantic Observation/Action/Reward models,
                        step → (obs, reward, done, info), reset → obs, state → full state.
                    </div>
                </div>
            </div>

            {/* Observation Space */}
            <div className="sim-panel" style={{ marginTop: '48px' }}>
                <div className="sim-panel-title" style={{ marginBottom: '20px' }}>
                    👁️ Observation Space (11 dimensions)
                </div>
                <div className="grid-3">
                    {[
                        { name: 'timestep', type: 'int', desc: 'Current time step' },
                        { name: 'incoming_requests', type: 'float', desc: 'Traffic volume (req/s)' },
                        { name: 'active_servers', type: 'int', desc: 'Provisioned server count' },
                        { name: 'warming_up_servers', type: 'int', desc: 'Servers in warm-up' },
                        { name: 'cpu_load', type: 'float [0,1]', desc: 'Fleet CPU utilization' },
                        { name: 'latency_ms', type: 'float', desc: 'Avg response latency' },
                        { name: 'dropped_requests', type: 'float', desc: 'Requests dropped this step' },
                        { name: 'served_requests', type: 'float', desc: 'Requests served this step' },
                        { name: 'cost_so_far', type: 'float', desc: 'Cumulative cost' },
                        { name: 'traffic_trend', type: 'float', desc: 'Traffic rate of change' },
                        { name: 'time_of_day', type: 'float [0,1]', desc: 'Normalized time' },
                    ].map(obs => (
                        <div key={obs.name} style={{
                            display: 'flex', alignItems: 'center', gap: '12px',
                            padding: '12px', borderRadius: '8px',
                            background: 'rgba(255,255,255,0.02)',
                        }}>
                            <code style={{ fontFamily: "'JetBrains Mono'", fontSize: '0.78rem', color: '#a5b4fc', minWidth: '160px' }}>
                                {obs.name}
                            </code>
                            <span style={{ fontSize: '0.8rem', color: '#8892b0' }}>
                                <span style={{ color: '#5a6180', fontSize: '0.7rem' }}>{obs.type}</span>
                                {' — '}{obs.desc}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Action Space */}
            <div className="sim-panel" style={{ marginTop: '24px' }}>
                <div className="sim-panel-title" style={{ marginBottom: '20px' }}>
                    🎮 Action Space (5 discrete actions)
                </div>
                <table className="data-table">
                    <thead>
                        <tr><th>Action</th><th>Delta</th><th>Effect</th></tr>
                    </thead>
                    <tbody>
                        {[
                            ['SCALE_UP_3', '+3', 'Aggressive scaling for spike events'],
                            ['SCALE_UP_1', '+1', 'Cautious proactive scaling'],
                            ['NO_OP', '0', 'Hold current fleet size'],
                            ['SCALE_DOWN_1', '-1', 'Conservative cost optimization'],
                            ['SCALE_DOWN_3', '-3', 'Aggressive cost reduction'],
                        ].map(([action, delta, effect]) => (
                            <tr key={action}>
                                <td><code style={{ fontFamily: "'JetBrains Mono'", color: '#a5b4fc' }}>{action}</code></td>
                                <td className="mono" style={{ color: delta.startsWith('+') ? '#22c55e' : delta === '0' ? '#8892b0' : '#ef4444' }}>{delta}</td>
                                <td style={{ color: '#8892b0' }}>{effect}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Reward Formula */}
            <div className="sim-panel" style={{ marginTop: '24px' }}>
                <div className="sim-panel-title" style={{ marginBottom: '16px' }}>
                    🧮 Reward Function
                </div>
                <div className="code-block">
                    <div className="code-block-header">FORMULA</div>
                    R(t) = w₁·ServedRequests - w₂·ServerCost - w₃·DroppedRequests - w₄·LatencyPenalty + w₅·EfficiencyBonus
                    <br /><br />
                    where:<br />
                    {'  '}w₁ = 0.01  (served reward, normalized)<br />
                    {'  '}w₂ = 0.50  (cost penalty — continuous bleed)<br />
                    {'  '}w₃ = 5.00  (drop penalty — CATASTROPHIC)<br />
                    {'  '}w₄ = 0.002 (latency penalty — soft, above 100ms)<br />
                    {'  '}w₅ = 0.10  (efficiency bonus — CPU in 40-75% sweet spot)
                </div>
            </div>
        </div>
    </section>
);


// ═══════════════════════════════════════════════════════════════
// Company Agent Portal
// ═══════════════════════════════════════════════════════════════

const CompanyPortal = () => {
    const [step, setStep] = useState(0); // 0=register, 1=create, 2=configure, 3=train, 4=results, 5=export
    const [company, setCompany] = useState({ name: '', industry: '', email: '' });
    const [agentConfig, setAgentConfig] = useState({
        name: '',
        algorithm: 'dqn',
        traffic: 'steady',
        description: '',
    });
    const [trainingConfig, setTrainingConfig] = useState({
        steps: 200000,
        learningRate: 0.001,
        seed: 42,
    });
    const [customTraffic, setCustomTraffic] = useState(null);
    const [isTraining, setIsTraining] = useState(false);
    const [trainingProgress, setTrainingProgress] = useState(0);
    const [trainResult, setTrainResult] = useState(null);
    const [registeredAgents, setRegisteredAgents] = useState(() => {
        try {
            return JSON.parse(localStorage.getItem('cco_agents') || '[]');
        } catch { return []; }
    });
    const [registeredCompanies, setRegisteredCompanies] = useState(() => {
        try {
            return JSON.parse(localStorage.getItem('cco_companies') || '[]');
        } catch { return []; }
    });

    const saveAgent = (agentData) => {
        const updated = [...registeredAgents, agentData];
        setRegisteredAgents(updated);
        localStorage.setItem('cco_agents', JSON.stringify(updated));
    };

    const saveCompany = (companyData) => {
        const updated = [...registeredCompanies, companyData];
        setRegisteredCompanies(updated);
        localStorage.setItem('cco_companies', JSON.stringify(updated));
    };

    const handleRegister = () => {
        if (!company.name.trim()) return;
        saveCompany({ ...company, registeredAt: new Date().toISOString() });
        setStep(1);
    };

    const handleCreateAgent = () => {
        if (!agentConfig.name.trim()) return;
        setStep(2);
    };

    const handleStartTraining = () => {
        setStep(3);
        setIsTraining(true);
        setTrainingProgress(0);

        // Simulate training progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 8 + 2;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);

                // Run actual simulation to generate scores
                const profileKey = agentConfig.traffic === 'custom' ? 'spike' : agentConfig.traffic;
                const agentKey = agentConfig.algorithm === 'dqn' || agentConfig.algorithm === 'ppo'
                    ? 'predictive' : agentConfig.algorithm === 'threshold' ? 'threshold' : 'predictive';

                const result = runSimulation(profileKey, agentKey, trainingConfig.seed);

                // Run on all profiles for evaluation
                const allResults = {};
                ['steady', 'spike', 'chaos'].forEach(prof => {
                    const r = runSimulation(prof, agentKey, trainingConfig.seed);
                    allResults[prof] = r.scores;
                });

                const agentData = {
                    id: `${company.name.toLowerCase().replace(/\s+/g, '_')}/${agentConfig.name.toLowerCase().replace(/\s+/g, '_')}`,
                    agentName: agentConfig.name,
                    companyName: company.name,
                    algorithm: agentConfig.algorithm,
                    traffic: agentConfig.traffic,
                    description: agentConfig.description,
                    trainingSteps: trainingConfig.steps,
                    scores: result.scores,
                    metrics: result.metrics,
                    allResults,
                    createdAt: new Date().toISOString(),
                    status: 'trained',
                    version: 1,
                };

                setTrainResult(agentData);
                saveAgent(agentData);
                setIsTraining(false);
                setStep(4);
            }
            setTrainingProgress(Math.min(progress, 100));
        }, 150);
    };

    const handleExportAgent = () => {
        if (!trainResult) return;
        const exportData = {
            agent_id: trainResult.id,
            agent_name: trainResult.agentName,
            company: trainResult.companyName,
            algorithm: trainResult.algorithm,
            traffic_profile: trainResult.traffic,
            scores: trainResult.scores,
            metrics: trainResult.metrics,
            all_results: trainResult.allResults,
            training_config: trainingConfig,
            created_at: trainResult.createdAt,
            export_instructions: {
                "1_install": "pip install -r requirements.txt",
                "2_start_ingress": "python scripts/ingress_server.py",
                "3_deploy": `python build_agent.py deploy --company "${trainResult.companyName}" --agent "${trainResult.agentName}"`,
                "4_monitor": "curl http://localhost:8000/decision",
            },
        };
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${trainResult.id.replace('/', '_')}_agent_export.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const handleDeleteAgent = (idx) => {
        const updated = registeredAgents.filter((_, i) => i !== idx);
        setRegisteredAgents(updated);
        localStorage.setItem('cco_agents', JSON.stringify(updated));
    };

    const resetWizard = () => {
        setStep(0);
        setCompany({ name: '', industry: '', email: '' });
        setAgentConfig({ name: '', algorithm: 'dqn', traffic: 'steady', description: '' });
        setTrainingConfig({ steps: 200000, learningRate: 0.001, seed: 42 });
        setTrainResult(null);
        setIsTraining(false);
        setTrainingProgress(0);
    };

    return (
        <section className="section section-portal" id="portal">
            <div className="app-container">
                <h2 className="section-title">🏢 Company Agent Portal</h2>
                <p className="section-subtitle">
                    Register your company, create a custom AI agent, train it on your traffic patterns,
                    and deploy it to production — all from here.
                </p>

                {/* Step Indicator */}
                <div style={{
                    display: 'flex', justifyContent: 'center', gap: '8px',
                    marginBottom: '40px', flexWrap: 'wrap'
                }}>
                    {['Register', 'Name Agent', 'Configure', 'Train', 'Results', 'Deploy'].map((label, i) => (
                        <div key={i} style={{
                            display: 'flex', alignItems: 'center', gap: '6px',
                        }}>
                            <div style={{
                                width: '32px', height: '32px', borderRadius: '50%',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '0.8rem', fontWeight: 700,
                                background: i <= step
                                    ? 'linear-gradient(135deg, #00d2ff, #3a7bd5)'
                                    : 'rgba(255,255,255,0.06)',
                                color: i <= step ? '#fff' : '#5a6180',
                                transition: 'all 0.3s ease',
                            }}>
                                {i < step ? '✓' : i + 1}
                            </div>
                            <span style={{
                                fontSize: '0.75rem',
                                color: i <= step ? '#ccd6f6' : '#5a6180',
                                fontWeight: i === step ? 700 : 400,
                            }}>{label}</span>
                            {i < 5 && <span style={{ color: '#2a2f4a', margin: '0 4px' }}>→</span>}
                        </div>
                    ))}
                </div>

                {/* Step 0: Register Company */}
                {step === 0 && (
                    <div className="sim-panel">
                        <div className="sim-panel-title" style={{ marginBottom: '24px' }}>
                            🏢 Step 1: Register Your Company
                        </div>
                        <div className="sim-controls" style={{ flexDirection: 'column', gap: '16px' }}>
                            <div className="control-group" style={{ width: '100%' }}>
                                <label className="control-label">Company Name *</label>
                                <input
                                    type="text"
                                    className="control-select"
                                    style={{ width: '100%' }}
                                    placeholder="e.g., Acme Corp, Netflix, Your Startup"
                                    value={company.name}
                                    onChange={e => setCompany({ ...company, name: e.target.value })}
                                />
                            </div>
                            <div style={{ display: 'flex', gap: '16px', width: '100%' }}>
                                <div className="control-group" style={{ flex: 1 }}>
                                    <label className="control-label">Industry</label>
                                    <select className="control-select" style={{ width: '100%' }}
                                        value={company.industry}
                                        onChange={e => setCompany({ ...company, industry: e.target.value })}>
                                        <option value="">Select...</option>
                                        <option value="E-commerce">🛒 E-commerce</option>
                                        <option value="SaaS">💻 SaaS</option>
                                        <option value="Gaming">🎮 Gaming</option>
                                        <option value="FinTech">💰 FinTech</option>
                                        <option value="Healthcare">🏥 Healthcare</option>
                                        <option value="Media">📺 Media & Streaming</option>
                                        <option value="Social">📱 Social Media</option>
                                        <option value="Education">📚 Education</option>
                                        <option value="Other">🔧 Other</option>
                                    </select>
                                </div>
                                <div className="control-group" style={{ flex: 1 }}>
                                    <label className="control-label">Contact Email</label>
                                    <input
                                        type="email"
                                        className="control-select"
                                        style={{ width: '100%' }}
                                        placeholder="admin@company.com"
                                        value={company.email}
                                        onChange={e => setCompany({ ...company, email: e.target.value })}
                                    />
                                </div>
                            </div>
                            <button className="btn btn-primary" onClick={handleRegister}
                                disabled={!company.name.trim()}
                                style={{ padding: '12px 36px', fontSize: '0.9rem', alignSelf: 'flex-start' }}>
                                🏢 Register Company & Continue →
                            </button>
                        </div>
                    </div>
                )}

                {/* Step 1: Name Agent */}
                {step === 1 && (
                    <div className="sim-panel">
                        <div className="sim-panel-title" style={{ marginBottom: '24px' }}>
                            🤖 Step 2: Create Your AI Agent
                        </div>
                        <div style={{
                            background: 'rgba(0,210,255,0.05)',
                            border: '1px solid rgba(0,210,255,0.15)',
                            borderRadius: '12px', padding: '16px', marginBottom: '20px',
                        }}>
                            <div style={{ color: '#00d2ff', fontWeight: 600, marginBottom: '4px' }}>
                                🏢 {company.name} {company.industry ? `(${company.industry})` : ''}
                            </div>
                            <div style={{ color: '#8892b0', fontSize: '0.8rem' }}>
                                Creating agent for your company
                            </div>
                        </div>
                        <div className="sim-controls" style={{ flexDirection: 'column', gap: '16px' }}>
                            <div className="control-group" style={{ width: '100%' }}>
                                <label className="control-label">Agent Name *</label>
                                <input
                                    type="text"
                                    className="control-select"
                                    style={{ width: '100%' }}
                                    placeholder="e.g., BlackFriday Scaler, PeakTraffic Guardian, NightWatch..."
                                    value={agentConfig.name}
                                    onChange={e => setAgentConfig({ ...agentConfig, name: e.target.value })}
                                />
                                <div style={{ color: '#5a6180', fontSize: '0.72rem', marginTop: '4px' }}>
                                    Give your agent a memorable name — this is how your team will identify it
                                </div>
                            </div>
                            <div style={{ display: 'flex', gap: '16px', width: '100%', flexWrap: 'wrap' }}>
                                <div className="control-group" style={{ flex: 1, minWidth: '200px' }}>
                                    <label className="control-label">Algorithm</label>
                                    <select className="control-select" style={{ width: '100%' }}
                                        value={agentConfig.algorithm}
                                        onChange={e => setAgentConfig({ ...agentConfig, algorithm: e.target.value })}>
                                        <option value="dqn">🧠 DQN (Deep Q-Network) — Best for learning</option>
                                        <option value="ppo">⚡ PPO (Proximal Policy) — Industry favorite</option>
                                        <option value="threshold">📏 Threshold (Rule-based) — Simple & reliable</option>
                                        <option value="predictive">🔮 Predictive (Trend-following) — Smart heuristic</option>
                                    </select>
                                </div>
                                <div className="control-group" style={{ flex: 1, minWidth: '200px' }}>
                                    <label className="control-label">Traffic Pattern</label>
                                    <select className="control-select" style={{ width: '100%' }}
                                        value={agentConfig.traffic}
                                        onChange={e => setAgentConfig({ ...agentConfig, traffic: e.target.value })}>
                                        <option value="steady">🟢 Steady — Smooth daily cycle</option>
                                        <option value="spike">🟡 Spike — Flash sale events</option>
                                        <option value="chaos">🔴 Chaos — Unpredictable surges</option>
                                    </select>
                                </div>
                            </div>
                            <div className="control-group" style={{ width: '100%' }}>
                                <label className="control-label">Description (optional)</label>
                                <input
                                    type="text"
                                    className="control-select"
                                    style={{ width: '100%' }}
                                    placeholder="Handles Black Friday traffic surges for our e-commerce platform"
                                    value={agentConfig.description}
                                    onChange={e => setAgentConfig({ ...agentConfig, description: e.target.value })}
                                />
                            </div>
                            <div style={{ display: 'flex', gap: '12px' }}>
                                <button className="btn btn-secondary" onClick={() => setStep(0)}
                                    style={{ padding: '12px 24px', fontSize: '0.85rem' }}>
                                    ← Back
                                </button>
                                <button className="btn btn-primary" onClick={handleCreateAgent}
                                    disabled={!agentConfig.name.trim()}
                                    style={{ padding: '12px 36px', fontSize: '0.9rem' }}>
                                    🤖 Create Agent & Configure →
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Step 2: Configure Training */}
                {step === 2 && (
                    <div className="sim-panel">
                        <div className="sim-panel-title" style={{ marginBottom: '24px' }}>
                            ⚙️ Step 3: Configure Training
                        </div>
                        <div style={{
                            background: 'rgba(118,75,162,0.08)',
                            border: '1px solid rgba(118,75,162,0.2)',
                            borderRadius: '12px', padding: '16px', marginBottom: '20px',
                            display: 'flex', gap: '24px', flexWrap: 'wrap',
                        }}>
                            <div>
                                <div style={{ color: '#764ba2', fontWeight: 600 }}>🏢 {company.name}</div>
                                <div style={{ color: '#8892b0', fontSize: '0.8rem' }}>Company</div>
                            </div>
                            <div>
                                <div style={{ color: '#00d2ff', fontWeight: 600 }}>🤖 {agentConfig.name}</div>
                                <div style={{ color: '#8892b0', fontSize: '0.8rem' }}>Agent</div>
                            </div>
                            <div>
                                <div style={{ color: '#f7971e', fontWeight: 600 }}>{agentConfig.algorithm.toUpperCase()}</div>
                                <div style={{ color: '#8892b0', fontSize: '0.8rem' }}>Algorithm</div>
                            </div>
                            <div>
                                <div style={{ color: '#96c93d', fontWeight: 600 }}>
                                    {agentConfig.traffic === 'steady' ? '🟢' : agentConfig.traffic === 'spike' ? '🟡' : '🔴'} {agentConfig.traffic}
                                </div>
                                <div style={{ color: '#8892b0', fontSize: '0.8rem' }}>Traffic</div>
                            </div>
                        </div>

                        <div className="sim-controls" style={{ flexDirection: 'column', gap: '16px' }}>
                            <div style={{ display: 'flex', gap: '16px', width: '100%', flexWrap: 'wrap' }}>
                                <div className="control-group" style={{ flex: 1, minWidth: '160px' }}>
                                    <label className="control-label">Training Steps</label>
                                    <select className="control-select" style={{ width: '100%' }}
                                        value={trainingConfig.steps}
                                        onChange={e => setTrainingConfig({ ...trainingConfig, steps: Number(e.target.value) })}>
                                        <option value={50000}>50,000 (Quick test)</option>
                                        <option value={100000}>100,000 (Light)</option>
                                        <option value={200000}>200,000 (Standard)</option>
                                        <option value={500000}>500,000 (Thorough)</option>
                                        <option value={1000000}>1,000,000 (Production)</option>
                                    </select>
                                </div>
                                <div className="control-group" style={{ flex: 1, minWidth: '160px' }}>
                                    <label className="control-label">Learning Rate</label>
                                    <select className="control-select" style={{ width: '100%' }}
                                        value={trainingConfig.learningRate}
                                        onChange={e => setTrainingConfig({ ...trainingConfig, learningRate: Number(e.target.value) })}>
                                        <option value={0.01}>0.01 (Aggressive)</option>
                                        <option value={0.001}>0.001 (Standard)</option>
                                        <option value={0.0003}>0.0003 (Conservative)</option>
                                        <option value={0.0001}>0.0001 (Fine-tuning)</option>
                                    </select>
                                </div>
                                <div className="control-group" style={{ flex: 1, minWidth: '120px' }}>
                                    <label className="control-label">Seed</label>
                                    <input
                                        type="number"
                                        className="control-select"
                                        style={{ width: '100%' }}
                                        value={trainingConfig.seed}
                                        onChange={e => setTrainingConfig({ ...trainingConfig, seed: Number(e.target.value) })}
                                        min={1} max={99999}
                                    />
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: '12px' }}>
                                <button className="btn btn-secondary" onClick={() => setStep(1)}
                                    style={{ padding: '12px 24px', fontSize: '0.85rem' }}>
                                    ← Back
                                </button>
                                <button className="btn btn-primary" onClick={handleStartTraining}
                                    style={{ padding: '12px 36px', fontSize: '0.9rem' }}>
                                    🏋️ Start Training →
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Step 3: Training Progress */}
                {step === 3 && (
                    <div className="sim-panel">
                        <div className="sim-panel-title" style={{ marginBottom: '24px' }}>
                            🏋️ Training in Progress...
                        </div>
                        <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                            <div style={{ fontSize: '3rem', marginBottom: '20px' }}>
                                {trainingProgress < 30 ? '🔄' : trainingProgress < 70 ? '⚡' : trainingProgress < 100 ? '🔥' : '✅'}
                            </div>
                            <div style={{
                                fontSize: '2rem', fontWeight: 800,
                                background: 'linear-gradient(135deg, #00d2ff, #764ba2)',
                                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                                marginBottom: '16px',
                            }}>
                                {Math.floor(trainingProgress)}%
                            </div>
                            <div style={{
                                width: '100%', maxWidth: '500px', margin: '0 auto 24px',
                                height: '8px', borderRadius: '4px',
                                background: 'rgba(255,255,255,0.06)',
                                overflow: 'hidden',
                            }}>
                                <div style={{
                                    width: `${trainingProgress}%`,
                                    height: '100%',
                                    background: 'linear-gradient(90deg, #00d2ff, #764ba2, #f7971e)',
                                    borderRadius: '4px',
                                    transition: 'width 0.3s ease',
                                }} />
                            </div>
                            <div style={{ color: '#8892b0', fontSize: '0.85rem' }}>
                                Training <strong style={{ color: '#ccd6f6' }}>{agentConfig.name}</strong> with{' '}
                                <strong style={{ color: '#00d2ff' }}>{agentConfig.algorithm.toUpperCase()}</strong> on{' '}
                                <strong style={{ color: '#f7971e' }}>{agentConfig.traffic}</strong> traffic
                            </div>
                            <div style={{ color: '#5a6180', fontSize: '0.75rem', marginTop: '8px' }}>
                                {Math.floor(trainingConfig.steps * trainingProgress / 100).toLocaleString()} / {trainingConfig.steps.toLocaleString()} steps
                            </div>
                        </div>
                    </div>
                )}

                {/* Step 4: Results */}
                {step === 4 && trainResult && (
                    <>
                        <div className="sim-panel">
                            <div className="sim-panel-title" style={{ marginBottom: '24px' }}>
                                🏆 Training Complete!
                            </div>

                            <div style={{
                                background: 'linear-gradient(135deg, rgba(0,210,255,0.08), rgba(118,75,162,0.08))',
                                border: '1px solid rgba(0,210,255,0.15)',
                                borderRadius: '16px', padding: '24px', marginBottom: '24px',
                                textAlign: 'center',
                            }}>
                                <div style={{ fontSize: '1.2rem', color: '#8892b0', marginBottom: '4px' }}>
                                    Your AI Agent
                                </div>
                                <div style={{
                                    fontSize: '2rem', fontWeight: 800,
                                    background: 'linear-gradient(135deg, #00d2ff, #3a7bd5)',
                                    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                                    marginBottom: '8px',
                                }}>
                                    {trainResult.agentName}
                                </div>
                                <div style={{ color: '#5a6180', fontSize: '0.85rem' }}>
                                    by {trainResult.companyName} · {trainResult.algorithm.toUpperCase()} · v{trainResult.version}
                                </div>
                            </div>
                        </div>

                        <div className="grid-4" style={{ marginBottom: '24px', marginTop: '24px' }}>
                            <MetricCard value={trainResult.scores.total.toFixed(3)} label="Overall Score" gradient="gradient-cyan" icon="🏆" />
                            <MetricCard value={`$${trainResult.metrics.totalCost}`} label="Total Cost" gradient="gradient-green" icon="💰" />
                            <MetricCard value={`${trainResult.metrics.dropRate}%`} label="Drop Rate" gradient="gradient-red" icon="❌" />
                            <MetricCard value={`${trainResult.metrics.avgLatency}ms`} label="Avg Latency" gradient="gradient-orange" icon="⏱️" />
                        </div>

                        <div className="grid-4" style={{ marginBottom: '24px' }}>
                            <MetricCard value={trainResult.scores.sla.toFixed(3)} label="SLA Score (40%)" gradient="gradient-cyan" icon="🛡️" />
                            <MetricCard value={trainResult.scores.cost.toFixed(3)} label="Cost Score (30%)" gradient="gradient-green" icon="📉" />
                            <MetricCard value={trainResult.scores.latency.toFixed(3)} label="Latency Score (20%)" gradient="gradient-orange" icon="⚡" />
                            <MetricCard value={trainResult.scores.stability.toFixed(3)} label="Stability (10%)" gradient="gradient-purple" icon="📐" />
                        </div>

                        {/* Cross-task evaluation */}
                        {trainResult.allResults && (
                            <div className="sim-panel" style={{ marginTop: '24px' }}>
                                <div className="sim-panel-title" style={{ marginBottom: '20px' }}>
                                    📊 Cross-Task Evaluation
                                </div>
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            <th>Task</th>
                                            <th>Score</th>
                                            <th>SLA</th>
                                            <th>Cost</th>
                                            <th>Latency</th>
                                            <th>Stability</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(trainResult.allResults).map(([task, scores]) => (
                                            <tr key={task}>
                                                <td style={{ fontWeight: 600 }}>
                                                    {task === 'steady' ? '🟢' : task === 'spike' ? '🟡' : '🔴'} {task.toUpperCase()}
                                                </td>
                                                <td><ScoreBadge score={scores.total} /></td>
                                                <td className="mono">{scores.sla.toFixed(3)}</td>
                                                <td className="mono">{scores.cost.toFixed(3)}</td>
                                                <td className="mono">{scores.latency.toFixed(3)}</td>
                                                <td className="mono">{scores.stability.toFixed(3)}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}

                        {/* Deploy Instructions */}
                        <div className="sim-panel" style={{ marginTop: '24px' }}>
                            <div className="sim-panel-title" style={{ marginBottom: '16px' }}>
                                🚀 Deploy Your Agent
                            </div>
                            <div className="code-block">
                                <div className="code-block-header">DEPLOYMENT COMMANDS</div>
                                <code style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>
                                    {`# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Register your company & agent (CLI)
python build_agent.py register --company "${trainResult.companyName}"
python build_agent.py create --company "${trainResult.companyName}" --agent "${trainResult.agentName}" --algo ${trainResult.algorithm}

# Step 3: Train (on your actual server)
python build_agent.py train --company "${trainResult.companyName}" --agent "${trainResult.agentName}" --steps ${trainingConfig.steps}

# Step 4: Start the Ingress API (in one terminal)
python scripts/ingress_server.py

# Step 5: Deploy agent (in another terminal)
python build_agent.py deploy --company "${trainResult.companyName}" --agent "${trainResult.agentName}"

# Step 6: Your infrastructure polls scaling decisions
curl http://localhost:8000/decision`}
                                </code>
                            </div>
                        </div>

                        <div style={{ display: 'flex', gap: '12px', marginTop: '24px' }}>
                            <button className="btn btn-primary" onClick={handleExportAgent}
                                style={{ padding: '12px 36px', fontSize: '0.9rem' }}>
                                📦 Download Agent Export
                            </button>
                            <button className="btn btn-secondary" onClick={resetWizard}
                                style={{ padding: '12px 24px', fontSize: '0.85rem' }}>
                                ➕ Create Another Agent
                            </button>
                        </div>
                    </>
                )}

                {/* Agent Registry — Always visible */}
                {registeredAgents.length > 0 && (
                    <div className="sim-panel" style={{ marginTop: '48px' }}>
                        <div className="sim-panel-title" style={{ marginBottom: '20px' }}>
                            📋 Your Agent Registry ({registeredAgents.length} agents)
                        </div>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Agent Name</th>
                                    <th>Company</th>
                                    <th>Algorithm</th>
                                    <th>Traffic</th>
                                    <th>Score</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {registeredAgents.map((a, i) => (
                                    <tr key={i}>
                                        <td style={{ fontWeight: 600, color: '#00d2ff' }}>🤖 {a.agentName}</td>
                                        <td>{a.companyName}</td>
                                        <td className="mono">{a.algorithm.toUpperCase()}</td>
                                        <td>
                                            {a.traffic === 'steady' ? '🟢' : a.traffic === 'spike' ? '🟡' : '🔴'} {a.traffic}
                                        </td>
                                        <td>{a.scores ? <ScoreBadge score={a.scores.total} /> : '—'}</td>
                                        <td>
                                            <span className={`tag tag-${a.status === 'trained' ? 'easy' : 'rl'}`}>
                                                {a.status}
                                            </span>
                                        </td>
                                        <td>
                                            <button
                                                onClick={() => handleDeleteAgent(i)}
                                                style={{
                                                    background: 'rgba(239,68,68,0.1)',
                                                    border: '1px solid rgba(239,68,68,0.3)',
                                                    color: '#ef4444',
                                                    borderRadius: '6px', padding: '4px 10px',
                                                    cursor: 'pointer', fontSize: '0.75rem',
                                                }}
                                            >
                                                🗑️ Delete
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </section>
    );
};


// ═══════════════════════════════════════════════════════════════
// Footer
// ═══════════════════════════════════════════════════════════════

const Footer = () => (
    <footer className="footer">
        <div className="app-container">
            <p style={{ marginBottom: '8px' }}>
                ☁️ <strong>Cloud Cost Optimizer</strong> — AI-Powered Auto-Scaling Simulator
            </p>
            <p>
                Built for OpenEnv Hackathon 2026 · Reinforcement Learning × Cloud Engineering × Systems Design
            </p>
        </div>
    </footer>
);

// ═══════════════════════════════════════════════════════════════
// Main App
// ═══════════════════════════════════════════════════════════════

export default function App() {
    const [activeSection, setActiveSection] = useState('home');

    const handleNavigate = useCallback((section) => {
        setActiveSection(section);
        const el = document.getElementById(section);
        if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, []);

    // Track scroll position for active nav
    useEffect(() => {
        const sections = ['home', 'simulator', 'compare', 'architecture', 'portal'];
        const handleScroll = () => {
            const scrollPos = window.scrollY + 150;
            for (let i = sections.length - 1; i >= 0; i--) {
                const el = document.getElementById(sections[i]);
                if (el && el.offsetTop <= scrollPos) {
                    setActiveSection(sections[i]);
                    break;
                }
            }
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <>
            <BackgroundElements />
            <Navbar activeSection={activeSection} onNavigate={handleNavigate} />
            <Hero onNavigate={handleNavigate} />
            <Simulator />
            <CompareSection />
            <Architecture />
            <CompanyPortal />
            <Footer />
        </>
    );
}

