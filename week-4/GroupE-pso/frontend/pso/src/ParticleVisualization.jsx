import { useRef, useEffect, useState } from 'react';

function ParticleVisualization({ data }) {
    const canvasRef = useRef(null);
    const [generationIndex, setGenerationIndex] = useState(0);
    const [speed, setSpeed] = useState(1);

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        const drawParticles = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const generationData = data[generationIndex];
            const particles = generationData.particles;
            const goalPosition = generationData.global_best_position;

            // Draw goal
            ctx.fillStyle = "red";
            ctx.beginPath();
            ctx.arc(goalPosition[0] * 10 + canvas.width / 2, goalPosition[1] * 10 + canvas.height / 2, 5, 0, 2 * Math.PI);
            ctx.fill();

            // Draw particles
            particles.forEach(([position, velocity]) => {
                ctx.fillStyle = "blue";
                ctx.beginPath();
                ctx.arc(position[0] * 10 + canvas.width / 2, position[1] * 10 + canvas.height / 2, 3, 0, 2 * Math.PI);
                ctx.fill();
            });
        };

        drawParticles();
    }, [data, generationIndex]);

    // Animation
    useEffect(() => {
        const interval = setInterval(() => {
            setGenerationIndex((prev) => (prev + 1) % data.length);
        }, 1000 / speed);

        return () => clearInterval(interval);
    }, [speed, data.length]);

    return (
        <div className="visualization">
            <canvas ref={canvasRef} width={600} height={400}></canvas>
            <div className="controls">
                <label>
                    Animation Speed:
                    <input type="range" min="1" max="10" value={speed} onChange={(e) => setSpeed(e.target.value)} />
                </label>
            </div>
        </div>
    );
}

export default ParticleVisualization;
