/**
 * Cloud Cost Optimizer — Simulation Engine (JavaScript)
 * 
 * A self-contained JS port of the Python environment so the React
 * frontend can run simulations entirely in the browser with no backend.
 */

// ─────────────────────────────────────────────────────────────────
// Traffic Generator
// ─────────────────────────────────────────────────────────────────

class SeededRandom {
    constructor(seed = 42) {
        this.seed = seed;
        this.state = seed;
    }

    next() {
        this.state = (this.state * 1664525 + 1013904223) & 0xffffffff;
        return (this.state >>> 0) / 0xffffffff;
    }

    gauss(mean = 0, std = 1) {
        const u1 = this.next();
        const u2 = this.next();
        const z = Math.sqrt(-2 * Math.log(u1 || 0.0001)) * Math.cos(2 * Math.PI * u2);
        return mean + z * std;
    }

    reset(seed) {
        this.seed = seed;
        this.state = seed;
    }
}

export function generateTraffic(config, totalSteps, seed = 42) {
    const rng = new SeededRandom(seed);
    const data = [];

    for (let t = 0; t < totalSteps; t++) {
        // Seasonal
        const seasonal = config.amplitude * Math.sin(
            2 * Math.PI * (t + config.phaseShift) / config.period
        );

        // Spikes
        let spikeTotal = 0;
        for (const spike of config.spikes) {
            if (t >= spike.start && t < spike.start + spike.duration) {
                const mid = spike.start + spike.duration / 2;
                const sigma = spike.duration / 4;
                spikeTotal += spike.magnitude * Math.exp(-0.5 * ((t - mid) / Math.max(sigma, 1)) ** 2);
            }
        }

        // Random burst
        let burst = 0;
        if (rng.next() < config.burstProb) {
            burst = (0.5 + rng.next() * 0.5) * config.burstMag;
        }

        // Noise
        const noise = rng.gauss(0, config.noiseStd);

        // Trend
        const trend = config.trend * t;

        const traffic = Math.max(config.minTraffic,
            config.baseLoad + seasonal + spikeTotal + burst + noise + trend
        );

        data.push(traffic);
    }

    return data;
}

// ─────────────────────────────────────────────────────────────────
// Traffic Profiles
// ─────────────────────────────────────────────────────────────────

export const TRAFFIC_PROFILES = {
    steady: {
        name: "Steady Traffic",
        difficulty: "easy",
        baseLoad: 100, amplitude: 30, period: 288, noiseStd: 5,
        spikes: [], trend: 0, burstProb: 0, burstMag: 0,
        phaseShift: 0, minTraffic: 5, maxSteps: 360, initialServers: 2,
    },
    spike: {
        name: "Spike Traffic",
        difficulty: "medium",
        baseLoad: 120, amplitude: 40, period: 288, noiseStd: 8,
        spikes: [
            { start: 50, duration: 30, magnitude: 200 },
            { start: 150, duration: 20, magnitude: 300 },
            { start: 250, duration: 40, magnitude: 250 },
        ],
        trend: 0, burstProb: 0.01, burstMag: 50,
        phaseShift: 0, minTraffic: 5, maxSteps: 360, initialServers: 3,
    },
    chaos: {
        name: "Chaos Traffic",
        difficulty: "hard",
        baseLoad: 150, amplitude: 60, period: 200, noiseStd: 20,
        spikes: [
            { start: 30, duration: 15, magnitude: 350 },
            { start: 80, duration: 25, magnitude: 200 },
            { start: 130, duration: 10, magnitude: 500 },
            { start: 200, duration: 35, magnitude: 300 },
            { start: 270, duration: 20, magnitude: 400 },
            { start: 330, duration: 15, magnitude: 250 },
        ],
        trend: 0.1, burstProb: 0.05, burstMag: 120,
        phaseShift: 0, minTraffic: 5, maxSteps: 400, initialServers: 3,
    },
};

// ─────────────────────────────────────────────────────────────────
// Server Fleet Model
// ─────────────────────────────────────────────────────────────────

export const SERVER_CONFIG = {
    capacityPerServer: 50,
    costPerServerPerStep: 0.1,
    warmupSteps: 3,
    baseLatency: 10,
    latencyGrowthFactor: 5,
    maxServers: 50,
    minServers: 1,
    slaLatencyThreshold: 200,
};

class ServerFleet {
    constructor(config = SERVER_CONFIG) {
        this.config = config;
        this.activeServers = config.minServers;
        this.warmingQueue = [];
    }

    get warmingCount() {
        return this.warmingQueue.length;
    }

    get totalCapacity() {
        return this.activeServers * this.config.capacityPerServer;
    }

    scale(delta) {
        if (delta > 0) {
            const canAdd = this.config.maxServers - this.activeServers - this.warmingCount;
            const actual = Math.min(delta, Math.max(0, canAdd));
            for (let i = 0; i < actual; i++) {
                this.warmingQueue.push(this.config.warmupSteps);
            }
            return actual;
        } else if (delta < 0) {
            const canRemove = this.activeServers - this.config.minServers;
            const actual = Math.min(Math.abs(delta), canRemove);
            this.activeServers -= actual;
            return -actual;
        }
        return 0;
    }

    tick() {
        const newQueue = [];
        for (let remaining of this.warmingQueue) {
            remaining--;
            if (remaining <= 0) {
                this.activeServers = Math.min(this.activeServers + 1, this.config.maxServers);
            } else {
                newQueue.push(remaining);
            }
        }
        this.warmingQueue = newQueue;
    }

    processRequests(incoming) {
        const capacity = this.totalCapacity;
        if (capacity <= 0) return { served: 0, dropped: incoming, cpuLoad: 1, latency: 10000 };

        const cpuLoad = Math.min(incoming / capacity, 1.0);
        const served = Math.min(incoming, capacity);
        const dropped = Math.max(0, incoming - capacity);
        const latency = Math.min(10000,
            this.config.baseLatency * Math.exp(cpuLoad * this.config.latencyGrowthFactor)
        );

        return { served, dropped, cpuLoad, latency };
    }

    getCost() {
        return this.activeServers * this.config.costPerServerPerStep +
            this.warmingCount * this.config.costPerServerPerStep * 0.5;
    }

    reset(initialServers = 1) {
        this.activeServers = Math.max(initialServers, this.config.minServers);
        this.warmingQueue = [];
    }
}

// ─────────────────────────────────────────────────────────────────
// Reward Function
// ─────────────────────────────────────────────────────────────────

export const REWARD_CONFIG = {
    wServed: 0.01,
    wCost: 0.5,
    wDropped: 5.0,
    wLatency: 0.002,
    wEfficiency: 0.1,
    latencyThreshold: 100,
    optimalCpuLow: 0.4,
    optimalCpuHigh: 0.75,
    maxRequestsNorm: 500,
};

function computeReward(served, dropped, cost, latency, cpuLoad, config = REWARD_CONFIG) {
    const servedReward = config.wServed * (served / Math.max(config.maxRequestsNorm, 1));
    const costPenalty = -config.wCost * cost;
    const dropPenalty = -config.wDropped * dropped;

    let latencyPenalty = 0;
    if (latency > config.latencyThreshold) {
        latencyPenalty = -config.wLatency * (latency - config.latencyThreshold);
    }

    let efficiencyBonus = 0;
    if (cpuLoad >= config.optimalCpuLow && cpuLoad <= config.optimalCpuHigh) {
        const mid = (config.optimalCpuLow + config.optimalCpuHigh) / 2;
        const closeness = 1 - Math.abs(cpuLoad - mid) / (config.optimalCpuHigh - config.optimalCpuLow);
        efficiencyBonus = config.wEfficiency * closeness;
    }

    return {
        total: servedReward + costPenalty + dropPenalty + latencyPenalty + efficiencyBonus,
        servedReward, costPenalty, dropPenalty, latencyPenalty, efficiencyBonus,
    };
}

// ─────────────────────────────────────────────────────────────────
// Agent Strategies
// ─────────────────────────────────────────────────────────────────

export const AGENTS = {
    static3: {
        name: "Static (3 Servers)",
        type: "baseline",
        act: (obs) => {
            if (obs.activeServers < 3) return 1;     // scale_up_1
            if (obs.activeServers > 3) return -1;    // scale_down_1
            return 0;
        },
    },
    static5: {
        name: "Static (5 Servers)",
        type: "baseline",
        act: (obs) => {
            if (obs.activeServers < 5) return 1;
            if (obs.activeServers > 5) return -1;
            return 0;
        },
    },
    threshold: {
        name: "Threshold (HPA-like)",
        type: "baseline",
        _lastScale: -100,
        act: (obs) => {
            if (obs.step - AGENTS.threshold._lastScale < 3) return 0;
            if (obs.cpuLoad > 0.75) {
                AGENTS.threshold._lastScale = obs.step;
                return obs.cpuLoad > 0.9 ? 3 : 1;
            }
            if (obs.cpuLoad < 0.30 && obs.activeServers > 1) {
                AGENTS.threshold._lastScale = obs.step;
                return -1;
            }
            return 0;
        },
    },
    predictive: {
        name: "Predictive Heuristic",
        type: "advanced",
        act: (obs) => {
            const neededServers = Math.max(1, Math.ceil(obs.incoming / 50));
            const total = obs.activeServers + obs.warmingServers;

            if (obs.trend > 5 && obs.cpuLoad > 0.65) {
                return total < neededServers + 2 ? 3 : 1;
            }
            if (obs.cpuLoad > 0.85) return 3;
            if (obs.cpuLoad > 0.65) return 1;
            if (obs.trend < -5 && obs.cpuLoad < 0.25 && obs.activeServers > neededServers + 1) return -1;
            if (obs.cpuLoad < 0.15 && obs.activeServers > 2) return -1;
            return 0;
        },
    },
    random: {
        name: "Random Agent",
        type: "baseline",
        act: () => [-3, -1, 0, 1, 3][Math.floor(Math.random() * 5)],
    },
};

// ─────────────────────────────────────────────────────────────────
// Full Simulation Runner
// ─────────────────────────────────────────────────────────────────

export function runSimulation(profileKey, agentKey, seed = 42) {
    const profile = TRAFFIC_PROFILES[profileKey];
    const agent = AGENTS[agentKey];
    const trafficData = generateTraffic(profile, profile.maxSteps, seed);

    const fleet = new ServerFleet();
    fleet.reset(profile.initialServers);

    // Reset threshold agent state
    if (AGENTS.threshold._lastScale !== undefined) AGENTS.threshold._lastScale = -100;

    const history = {
        traffic: [],
        servers: [],
        cpuLoad: [],
        latency: [],
        dropped: [],
        served: [],
        cost: [],
        reward: [],
        cumCost: [],
        cumDropped: [],
    };

    let totalCost = 0;
    let totalServed = 0;
    let totalDropped = 0;
    let totalReward = 0;
    let prevTraffic = trafficData[0];

    for (let step = 0; step < profile.maxSteps; step++) {
        const incoming = trafficData[step];
        const trend = step > 0 ? incoming - prevTraffic : 0;

        const obs = {
            step,
            incoming,
            activeServers: fleet.activeServers,
            warmingServers: fleet.warmingCount,
            cpuLoad: fleet.totalCapacity > 0 ? Math.min(incoming / fleet.totalCapacity, 1) : 1,
            trend,
        };

        // Agent decision
        const delta = agent.act(obs);
        fleet.scale(delta);
        fleet.tick();

        // Process
        const result = fleet.processRequests(incoming);
        const stepCost = fleet.getCost();
        const reward = computeReward(result.served, result.dropped, stepCost, result.latency, result.cpuLoad);

        totalCost += stepCost;
        totalServed += result.served;
        totalDropped += result.dropped;
        totalReward += reward.total;

        history.traffic.push(Math.round(incoming * 10) / 10);
        history.servers.push(fleet.activeServers);
        history.cpuLoad.push(Math.round(result.cpuLoad * 1000) / 1000);
        history.latency.push(Math.round(result.latency * 10) / 10);
        history.dropped.push(Math.round(result.dropped * 10) / 10);
        history.served.push(Math.round(result.served * 10) / 10);
        history.cost.push(Math.round(stepCost * 1000) / 1000);
        history.reward.push(Math.round(reward.total * 10000) / 10000);
        history.cumCost.push(Math.round(totalCost * 100) / 100);
        history.cumDropped.push(Math.round(totalDropped * 10) / 10);

        prevTraffic = incoming;
    }

    // Grading
    const totalRequests = trafficData.reduce((a, b) => a + b, 0);
    const dropRate = totalDropped / Math.max(totalRequests, 1);
    const avgLatency = history.latency.reduce((a, b) => a + b, 0) / history.latency.length;
    const maxLatency = Math.max(...history.latency);
    const avgCpu = history.cpuLoad.reduce((a, b) => a + b, 0) / history.cpuLoad.length;
    const slaCompliance = history.latency.filter(l => l <= 200).length / history.latency.length;
    const peakServers = Math.max(...history.servers);
    const avgServers = history.servers.reduce((a, b) => a + b, 0) / history.servers.length;

    // Score components
    let slaScore;
    if (dropRate <= 0) slaScore = 1;
    else if (dropRate <= 0.05) slaScore = 1 - (dropRate / 0.05) * 0.5;
    else slaScore = Math.max(0, 0.5 - (dropRate - 0.05) * 5);
    slaScore = 0.6 * slaScore + 0.4 * slaCompliance;

    const worstCost = 180;
    const bestCost = 36;
    let costScore;
    if (totalCost <= bestCost) costScore = 1;
    else if (totalCost >= worstCost) costScore = 0;
    else costScore = 1 - (totalCost - bestCost) / (worstCost - bestCost);

    let latencyScore;
    if (avgLatency <= 60) latencyScore = 1;
    else if (avgLatency <= 200) latencyScore = 1 - 0.5 * ((avgLatency - 60) / 140);
    else latencyScore = Math.max(0, 0.5 - (avgLatency - 200) / 400);
    let maxLatScore;
    if (maxLatency <= 200) maxLatScore = 1;
    else if (maxLatency <= 600) maxLatScore = 1 - (maxLatency - 200) / 400;
    else maxLatScore = 0;
    latencyScore = 0.7 * latencyScore + 0.3 * maxLatScore;

    // Stability
    let stabilityScore = 0.5;
    if (history.servers.length >= 2) {
        const changes = [];
        for (let i = 1; i < history.servers.length; i++) {
            changes.push(Math.abs(history.servers[i] - history.servers[i - 1]));
        }
        const avgChange = changes.reduce((a, b) => a + b, 0) / changes.length;
        if (avgChange <= 0.1) stabilityScore = 1;
        else if (avgChange <= 0.5) stabilityScore = 0.8;
        else if (avgChange <= 1) stabilityScore = 0.6;
        else stabilityScore = Math.max(0, 0.6 - (avgChange - 1) * 0.2);
    }

    const totalScore = Math.max(0, Math.min(1,
        0.40 * slaScore + 0.30 * costScore + 0.20 * latencyScore + 0.10 * stabilityScore
    ));

    return {
        history,
        metrics: {
            totalReward: Math.round(totalReward * 100) / 100,
            totalCost: Math.round(totalCost * 100) / 100,
            totalServed: Math.round(totalServed),
            totalDropped: Math.round(totalDropped),
            totalRequests: Math.round(totalRequests),
            dropRate: Math.round(dropRate * 10000) / 100,
            avgLatency: Math.round(avgLatency * 10) / 10,
            maxLatency: Math.round(maxLatency * 10) / 10,
            avgCpu: Math.round(avgCpu * 1000) / 10,
            slaCompliance: Math.round(slaCompliance * 10000) / 100,
            peakServers,
            avgServers: Math.round(avgServers * 10) / 10,
        },
        scores: {
            total: Math.round(totalScore * 10000) / 10000,
            sla: Math.round(slaScore * 1000) / 1000,
            cost: Math.round(costScore * 1000) / 1000,
            latency: Math.round(latencyScore * 1000) / 1000,
            stability: Math.round(stabilityScore * 1000) / 1000,
        },
        profile: profile.name,
        agent: agent.name,
    };
}

// Compare all agents on a given profile
export function compareAllAgents(profileKey, seed = 42) {
    const results = [];
    for (const [key, agent] of Object.entries(AGENTS)) {
        if (AGENTS.threshold._lastScale !== undefined) AGENTS.threshold._lastScale = -100;
        const result = runSimulation(profileKey, key, seed);
        results.push({ key, ...result });
    }
    return results.sort((a, b) => b.scores.total - a.scores.total);
}
