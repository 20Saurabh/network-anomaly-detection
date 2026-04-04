import React, { useState } from 'react';

const Upload = ({ onFileLoaded }) => {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleLoad = () => {
        if (!file) return;
        setLoading(true);
        // Pass raw file object to parent for client-side parsing
        onFileLoaded(file);
        setLoading(false);
    };

    return (
        <div className="card">
            <h3>Load Dataset</h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                Select a CSV file. It will be processed locally in your browser (no upload needed).
            </p>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <input
                    type="file"
                    onChange={handleFileChange}
                    className="input"
                    accept=".csv"
                />
                <button
                    onClick={handleLoad}
                    disabled={loading || !file}
                    className="btn btn-primary"
                >
                    {loading ? 'Loading...' : 'Load Data'}
                </button>
            </div>
        </div>
    );
};

export default Upload;
