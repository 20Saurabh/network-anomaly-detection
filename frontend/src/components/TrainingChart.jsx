import React from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const TrainingChart = ({ history }) => {
    const data = {
        labels: history.map((_, i) => i + 1),
        datasets: [
            {
                label: 'Training Loss',
                data: history,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.5)',
                tension: 0.3,
            },
        ],
    };

    const options = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
                labels: { color: '#94a3b8' }
            },
            title: {
                display: true,
                text: 'Model Training Progress',
                color: '#f8fafc'
            },
            tooltip: {
                mode: 'index',
                intersect: false,
            }
        },
        scales: {
            y: {
                grid: { color: '#334155' },
                ticks: { color: '#94a3b8' }
            },
            x: {
                grid: { color: '#334155' },
                ticks: { color: '#94a3b8' }
            }
        },
        animation: false
    };

    return (
        <div className="card" style={{ marginTop: '1.5rem' }}>
            <Line options={options} data={data} />
        </div>
    );
};

export default TrainingChart;
