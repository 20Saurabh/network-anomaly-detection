import React from 'react';
import {
    Chart as ChartJS,
    LinearScale,
    PointElement,
    LineElement, // for threshold line
    Tooltip,
    Legend,
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend);

const ResultsChart = ({ results, threshold }) => {
    if (!results || results.length === 0) return null;

    // Separate normal vs anomalies for coloring
    const normalPoints = results
        .map((r, i) => ({ x: i, y: r._score, _type: 'normal' }))
        .filter(p => p.y <= threshold);

    const anomalyPoints = results
        .map((r, i) => ({ x: i, y: r._score, _type: 'anomaly' }))
        .filter(p => p.y > threshold);

    const data = {
        datasets: [
            {
                label: 'Normal',
                data: normalPoints,
                backgroundColor: '#10b981', // Green
                pointRadius: 3,
                pointHoverRadius: 5,
            },
            {
                label: 'Anomaly',
                data: anomalyPoints,
                backgroundColor: '#ef4444', // Red
                pointRadius: 6,
                pointHoverRadius: 8,
            },
            // Threshold Line (Approximated as a dataset or annotation)
            // Since scatter charts x-axis is numeric, we can draw a line dataset?
            // Or just a line dataset
            {
                type: 'line',
                label: 'Threshold',
                data: [{ x: 0, y: threshold }, { x: results.length, y: threshold }],
                borderColor: '#f59e0b', // Amber/Orange
                borderWidth: 2,
                pointRadius: 0,
                borderDash: [5, 5],
                fill: false
            }

        ],
    };

    const options = {
        responsive: true,
        plugins: {
            legend: {
                labels: { color: '#94a3b8' }
            },
            title: {
                display: true,
                text: 'Anomaly Score Distribution',
                color: '#f8fafc'
            },
            tooltip: {
                callbacks: {
                    label: (ctx) => `Row ${ctx.raw.x}: Score ${ctx.raw.y.toFixed(4)}`
                }
            }
        },
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                title: { display: true, text: 'Row Index', color: '#64748b' },
                grid: { color: '#334155' },
                ticks: { color: '#94a3b8' }
            },
            y: {
                title: { display: true, text: 'Anomaly Score (MSE)', color: '#64748b' },
                grid: { color: '#334155' },
                ticks: { color: '#94a3b8' },
                min: 0,
            }
        }
    };

    return (
        <div className="card" style={{ marginTop: '1.5rem' }}>
            <Scatter options={options} data={data} />
        </div>
    );
};

export default ResultsChart;
