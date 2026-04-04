import React, { useState, useEffect, useRef } from 'react';
import Papa from 'papaparse';
import * as tf from '@tensorflow/tfjs';
import Upload from './Upload';
import TrainingChart from './TrainingChart';
import ResultsChart from './ResultsChart';
import { TimeSeriesWithAnomalies } from './AdvancedCharts';

const Dashboard = () => {
    const [data, setData] = useState([]);
    const [headers, setHeaders] = useState([]);
    const [lossHistory, setLossHistory] = useState([]);
    const [isTraining, setIsTraining] = useState(false);
    const [status, setStatus] = useState('System Ready');
    const [model, setModel] = useState(null);
    const [anomalies, setAnomalies] = useState([]);
    const [threshold, setThreshold] = useState(0);
    const [analysisComplete, setAnalysisComplete] = useState(false);

    // Feature Selection for Time Series
    const [selectedFeature, setSelectedFeature] = useState(null);

    // Initializer after headers are set
    useEffect(() => {
        if (headers.length > 0 && !selectedFeature) {
            const firstFeature = headers.find(h => h.toLowerCase() !== 'label' && h.toLowerCase() !== 'is_anomaly');
            if (firstFeature) setSelectedFeature(firstFeature);
        }
    }, [headers, selectedFeature]);
    // Model Config
    const [epochs, setEpochs] = useState(100); // Increased default
    const [learningRate, setLearningRate] = useState(0.01);
    const [sensitivity, setSensitivity] = useState(1.5); // Default lower for more alerts
    const stopTrainingRef = useRef(false);

    const handleFileLoaded = (file) => {
        setAnalysisComplete(false);
        setAnomalies([]);
        setLossHistory([]);

        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => {
                const rows = results.data;
                if (rows.length > 0) {
                    setData(rows);
                    setHeaders(Object.keys(rows[0]));
                    setStatus(`Dataset Loaded: ${rows.length} records.`);
                }
            }
        });
    };

    // Trigger "Fake" Backend Logs
    const triggerBackendSimulation = async () => {
        try {
            await fetch('http://127.0.0.1:8000/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: 'network_traffic.csv',
                    row_count: data.length
                })
            });
        } catch (e) {
            console.warn("Backend simulation server not reachable (this is fine for client-only mode)");
        }
    };

    const trainModel = async () => {
        if (data.length === 0) return;
        setIsTraining(true);
        setAnalysisComplete(false);
        setLossHistory([]);
        setAnomalies([]);

        stopTrainingRef.current = false;
        setStatus('Initializing Neural Network...');

        // Trigger simulation logs
        triggerBackendSimulation();

        // 1. Data Prep
        const featureCols = headers.filter(h => h.toLowerCase() !== 'label' && h.toLowerCase() !== 'is_anomaly');

        const tensorData = tf.tidy(() => {
            const rawData = data.map(row => featureCols.map(c => row[c]).filter(val => typeof val === 'number'));
            const tensor = tf.tensor2d(rawData);
            const min = tensor.min(0);
            const max = tensor.max(0);
            return tensor.sub(min).div(max.sub(min).add(1e-8));
        });

        // 2. Build Model
        const inputDim = tensorData.shape[1];
        const newModel = tf.sequential();
        newModel.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputDim] }));
        newModel.add(tf.layers.dense({ units: 8, activation: 'relu' }));
        newModel.add(tf.layers.dense({ units: 16, activation: 'relu' }));
        newModel.add(tf.layers.dense({ units: inputDim, activation: 'sigmoid' }));
        newModel.compile({ optimizer: tf.train.adam(learningRate), loss: 'meanSquaredError' });

        setModel(newModel);
        setStatus('Training Model...');

        // 3. Train
        await newModel.fit(tensorData, tensorData, {
            epochs: epochs,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (stopTrainingRef.current) {
                        newModel.stopTraining = true;
                    }
                    setLossHistory(prev => [...prev, logs.loss]);
                    setStatus(`Training Epoch ${epoch + 1}/${epochs} | Loss: ${logs.loss.toFixed(6)}`);
                }
            }
        });

        if (!stopTrainingRef.current) {
            setStatus('Analyzing Traffic Flows...');
            detectAnomalies(newModel, tensorData, featureCols);
        } else {
            setStatus('Training Halted.');
        }

        tensorData.dispose();
        setIsTraining(false);
    };

    const detectAnomalies = (trainedModel, tensorData, featureCols) => {
        tf.tidy(() => {
            const preds = trainedModel.predict(tensorData);

            // Calculate squared error per feature
            const squaredErrors = tensorData.sub(preds).square();
            const text = squaredErrors.mean(1);
            const scores = text.arraySync();
            const errorValues = squaredErrors.arraySync();

            // Threshold Calculation
            const n = scores.length;
            const mean = scores.reduce((a, b) => a + b, 0) / n;
            const stdDev = Math.sqrt(scores.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / n);
            const calculatedThreshold = mean + (sensitivity * stdDev);
            setThreshold(calculatedThreshold);

            const hasLabel = headers.some(h => h.toLowerCase() === 'label');
            const labelCol = headers.find(h => h.toLowerCase() === 'label');

            const results = data.map((row, i) => {
                const score = scores[i];
                const isAnomaly = score > calculatedThreshold;

                // Identify feature with max error for this row
                const rowErrors = errorValues[i];
                const maxErrorIndex = rowErrors.indexOf(Math.max(...rowErrors));
                const reasonFeature = featureCols[maxErrorIndex];
                const reasonValue = row[reasonFeature];

                let confidence = 0.5;
                if (isAnomaly) {
                    const diff = score - calculatedThreshold;
                    confidence = 0.5 + (0.5 * (1 - Math.exp(-diff / stdDev)));
                } else {
                    const diff = calculatedThreshold - score;
                    confidence = 0.5 + (0.5 * (1 - Math.exp(-diff / stdDev)));
                }

                return {
                    ...row,
                    _score: score,
                    _label: isAnomaly ? 1 : 0,
                    _isAnomaly: isAnomaly,
                    _confidence: Math.min(0.99, confidence),
                    _trueLabel: hasLabel ? row[labelCol] : null,
                    _reason: `${reasonFeature}: ${reasonValue}`
                };
            });

            setAnomalies(results);
            setAnalysisComplete(true);
            const anomalyCount = results.filter(r => r._isAnomaly).length;
            setStatus(`Analysis Complete. ${anomalyCount} Threats Detected.`);


        });
    };



    const stopTraining = () => {
        stopTrainingRef.current = true;
    };

    const flaggedCount = anomalies.filter(a => a._isAnomaly).length;

    return (
        <div className="container">
            <header style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h1>NIDS Dashboard</h1>
                    <p style={{ color: 'var(--text-secondary)' }}>Network Intrusion Detection System</p>
                </div>
                {analysisComplete && (
                    <div style={{
                        padding: '0.5rem 1rem',
                        borderRadius: '4px',
                        backgroundColor: flaggedCount > 0 ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                        color: flaggedCount > 0 ? '#ef4444' : '#10b981',
                        border: `1px solid ${flaggedCount > 0 ? '#ef4444' : '#10b981'}`,
                        fontWeight: 'bold'
                    }}>
                        {flaggedCount > 0 ? `${flaggedCount} THREATS DETECTED` : 'SYSTEM SECURE'}
                    </div>
                )}
            </header>

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1fr) 2fr', gap: '2rem' }}>
                {/* Left Column: Config */}
                <div>
                    <Upload onFileLoaded={handleFileLoaded} />

                    <div className="card" style={{ marginTop: '1.5rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <h3>Configuration</h3>
                            <span style={{ fontSize: '0.8rem', color: 'var(--accent)' }}>AUTOENCODER</span>
                        </div>
                        <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <div>
                                <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Epochs</label>
                                <input type="number" className="input" value={epochs} onChange={e => setEpochs(+e.target.value)} />
                            </div>
                            <div>
                                <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Learning Rate</label>
                                <input type="number" step="0.001" className="input" value={learningRate} onChange={e => setLearningRate(+e.target.value)} />
                            </div>
                            <div>
                                <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Sensitivity (StdDev)</label>
                                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                                    <input
                                        type="range"
                                        min="0.5"
                                        max="4.0"
                                        step="0.1"
                                        value={sensitivity}
                                        onChange={e => setSensitivity(+e.target.value)}
                                        style={{ flex: 1 }}
                                    />
                                    <span style={{ fontFamily: 'monospace', width: '30px' }}>{sensitivity.toFixed(1)}</span>
                                </div>
                            </div>
                        </div>

                        <div style={{ marginTop: '1.5rem', display: 'flex', gap: '0.5rem' }}>
                            <button
                                className="btn btn-primary"
                                style={{ flex: 1 }}
                                onClick={trainModel}
                                disabled={isTraining || data.length === 0}
                            >
                                {isTraining ? 'Training Model...' : (analysisComplete ? 'Retrain System' : 'Initialize Analysis')}
                            </button>
                            {isTraining && (
                                <button className="btn" style={{ backgroundColor: 'var(--danger)', color: 'white' }} onClick={stopTraining}>
                                    HALT
                                </button>
                            )}
                        </div>
                        <p style={{ marginTop: '1rem', fontWeight: 500, fontSize: '0.9rem', fontFamily: 'monospace' }}>{status}</p>
                    </div>

                    {/* Show simple stats during training */}
                    {isTraining && lossHistory.length > 0 && (
                        <div className="card" style={{ marginTop: '1.5rem' }}>
                            <h4>Current Loss</h4>
                            <p style={{ fontSize: '2rem', margin: 0, fontFamily: 'monospace' }}>{lossHistory[lossHistory.length - 1].toFixed(6)}</p>
                        </div>
                    )}


                </div>

                {/* Right Column: Visualization */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

                    {!analysisComplete ? (
                        /* Show Training Progress when NOT done */
                        lossHistory.length > 0 ? (
                            <TrainingChart history={lossHistory} />
                        ) : (
                            <div className="card" style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', borderStyle: 'dashed' }}>
                                <p style={{ color: 'var(--text-secondary)' }}>Awaiting Data Stream...</p>
                            </div>
                        )
                    ) : (
                        /* Show RESULT DASHBOARD when Analysis is Complete */
                        <>
                            {/* Summary Cards */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                                <div className="card" style={{ textAlign: 'center' }}>
                                    <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Total Flows</h4>
                                    <p style={{ fontSize: '1.8rem', margin: '0.5rem 0 0', fontWeight: 'bold' }}>{data.length}</p>
                                </div>
                                <div className="card" style={{ textAlign: 'center', border: '1px solid var(--danger)' }}>
                                    <h4 style={{ color: 'var(--danger)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Threats Identified</h4>
                                    <p style={{ fontSize: '1.8rem', margin: '0.5rem 0 0', color: 'var(--danger)', fontWeight: 'bold' }}>{flaggedCount}</p>
                                </div>
                                <div className="card" style={{ textAlign: 'center' }}>
                                    <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Peak Anomaly Score</h4>
                                    <p style={{ fontSize: '1.8rem', margin: '0.5rem 0 0', fontWeight: 'bold', fontFamily: 'monospace' }}>
                                        {Math.max(...anomalies.map(a => a._score)).toFixed(4)}
                                    </p>
                                </div>
                            </div>

                            {/* Scatter Visualization - Anomaly Score Over Time/Index */}
                            <ResultsChart results={anomalies} threshold={threshold} />




                            {/* Time Series Context - Feature Selector & Plot */}
                            {headers.length > 0 && selectedFeature && (
                                <div className="card" style={{ marginTop: '1.5rem' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                        <h3 style={{ margin: 0 }}>Feature Analysis</h3>
                                        <div>
                                            <label style={{ marginRight: '10px', color: 'var(--text-secondary)' }}>Select Feature:</label>
                                            <select
                                                className="input"
                                                style={{ padding: '0.25rem 0.5rem' }}
                                                value={selectedFeature}
                                                onChange={(e) => setSelectedFeature(e.target.value)}
                                            >
                                                {headers.filter(h => h.toLowerCase() !== 'label' && h.toLowerCase() !== 'is_anomaly').map(h => (
                                                    <option key={h} value={h}>{h}</option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>

                                    <TimeSeriesWithAnomalies
                                        data={anomalies}
                                        anomalies={anomalies}
                                        featureName={selectedFeature}
                                    />
                                    <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: 'rgba(99, 102, 241, 0.05)', borderRadius: '6px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                        <strong>Why isn't a big peak always an anomaly?</strong>
                                        <ul style={{ margin: '0.5rem 0 0 1.2rem', padding: 0 }}>
                                            <li>This AI model analyzes <strong>all features simultaneously</strong> (Multivariate).</li>
                                            <li>A high value (peak) in one feature might be "normal" if other features also increase (correlation).</li>
                                            <li>An anomaly occurs when the <strong>relationship</strong> between features breaks, even if values are small.</li>
                                        </ul>
                                    </div>
                                </div>
                            )}

                            {/* Detailed Table */}
                            <div className="card">
                                <h3>Flagged Threats</h3>
                                <div style={{ overflowX: 'auto', maxHeight: '400px', marginTop: '1rem' }}>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                                        <thead style={{ position: 'sticky', top: 0, backgroundColor: 'var(--bg-secondary)' }}>
                                            <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border)' }}>
                                                <th style={{ padding: '0.75rem 0.5rem' }}>ID</th>
                                                <th style={{ padding: '0.75rem 0.5rem', color: 'var(--danger)' }}>Score</th>
                                                <th style={{ padding: '0.75rem 0.5rem' }}>Confidence</th>
                                                <th style={{ padding: '0.75rem 0.5rem' }}>Reason</th>
                                                {headers.map(h => <th key={h} style={{ padding: '0.75rem 0.5rem' }}>{h}</th>)}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {anomalies.filter(r => r._isAnomaly).map((row, i) => (
                                                <tr key={i} style={{ borderBottom: '1px solid var(--border)', backgroundColor: 'rgba(239, 68, 68, 0.05)' }}>
                                                    <td style={{ padding: '0.5rem', color: 'var(--text-secondary)' }}>#{i}</td>
                                                    <td style={{ padding: '0.5rem', color: 'var(--danger)', fontWeight: 'bold', fontFamily: 'monospace' }}>{row._score.toFixed(4)}</td>
                                                    <td style={{ padding: '0.5rem' }}>{(row._confidence * 100).toFixed(1)}%</td>
                                                    <td style={{ padding: '0.5rem' }}>
                                                        <span style={{
                                                            padding: '2px 6px',
                                                            borderRadius: '4px',
                                                            backgroundColor: 'rgba(99, 102, 241, 0.2)',
                                                            color: '#818cf8',
                                                            fontSize: '0.75rem',
                                                            fontWeight: 'bold',
                                                            textTransform: 'uppercase'
                                                        }}>
                                                            {row._reason}
                                                        </span>
                                                    </td>
                                                    {headers.map(h => <td key={h} style={{ padding: '0.5rem', fontFamily: 'monospace', fontSize: '0.8rem' }}>{row[h]}</td>)}
                                                </tr>
                                            ))}
                                            {flaggedCount === 0 && (
                                                <tr>
                                                    <td colSpan={headers.length + 4} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
                                                        System Secure: No deviations detected above threshold ({threshold.toFixed(4)})
                                                    </td>
                                                </tr>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                            {/* Collapsible Training Chart */}
                            <div className="card" style={{ opacity: 0.8 }}>
                                <details>
                                    <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)' }}>View Training Telemetry</summary>
                                    <TrainingChart history={lossHistory} />
                                </details>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
