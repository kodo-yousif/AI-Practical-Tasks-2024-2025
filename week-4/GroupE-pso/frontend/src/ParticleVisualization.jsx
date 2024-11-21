import { useRef, useEffect, useState } from 'react';

function ParticleVisualization({ data }) {
    const canvasRef = useRef(null);
    const [generationIndex, setGenerationIndex] = useState(0);
    const [speed, setSpeed] = useState(1);

    const [zoom, setZoom] = useState(1);
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const canvasWidth = 700;
    const canvasHeight = 400;

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        const drawParticles = () => {
            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            ctx.translate(offset.x, offset.y);
            ctx.scale(zoom, zoom);

            const generationData = data[generationIndex];
            const { global_best, particles } = generationData;

            console.log("Particles:", particles);

            // Draw the global best position
            ctx.fillStyle = "white";
            ctx.beginPath();
            ctx.arc(
                global_best.position.x * 10 + canvas.width / 2,
                global_best.position.y * 10 + canvas.height / 2,
                15,
                0,
                2 * Math.PI
            );
            ctx.fill();

            // Draw particles
            particles.forEach((particle) => {
                const { position } = particle;
                ctx.fillStyle = "red";
                ctx.beginPath();
                ctx.arc(
                    position.x * 10 + canvas.width / 2,
                    position.y * 10 + canvas.height / 2,
                    8,
                    0,
                    2 * Math.PI
                );
                ctx.fill();
            });

            ctx.restore();
        };

        drawParticles();
    }, [generationIndex, data, zoom, offset]);


    useEffect(() => {
        const interval = setInterval(() => {
            setGenerationIndex((prev) => {
                if (prev + 1 < data.length) {
                    return prev + 1;
                } else {
                    clearInterval(interval);
                    return prev;
                }
            });
        }, 1000 / speed);

        return () => clearInterval(interval);
    }, [speed, data.length]);


    // canvas logic
    const calculateBounds = () => {
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        data.forEach((generation) => {
            generation.particles.forEach((particle) => {
                minX = Math.min(minX, particle.position.x);
                minY = Math.min(minY, particle.position.y);
                maxX = Math.max(maxX, particle.position.x);
                maxY = Math.max(maxY, particle.position.y);
            });
        });

        return { minX, minY, maxX, maxY };
    };

    const { minX, minY, maxX, maxY } = calculateBounds();

    const scaleFactor = Math.min(
        (canvasWidth * 0.8) / (maxX - minX),
        (canvasHeight * 0.8) / (maxY - minY)
    ) * zoom;

    const clamp = (value, min, max) => Math.min(Math.max(value, min), max);


    const handleWheel = (e) => {
        e.preventDefault();
        const zoomDelta = e.deltaY > 0 ? -0.1 : 0.1;
        const newZoom = clamp(zoom + zoomDelta, 0.5, 5); // Clamp zoom between 0.5 and 5
        setZoom(newZoom);
    };

    const handleMouseDown = (e) => {
        const rect = canvasRef.current.getBoundingClientRect();
        const startX = e.clientX - rect.left - offset.x / scaleFactor;
        const startY = e.clientY - rect.top - offset.y / scaleFactor;

        const handleMouseMove = (moveEvent) => {
            const newOffsetX = moveEvent.clientX - rect.left - startX * scaleFactor;
            const newOffsetY = moveEvent.clientY - rect.top - startY * scaleFactor;

            setOffset({
                x: clamp(newOffsetX, canvasWidth - scaleFactor * (maxX - minX), 0),
                y: clamp(newOffsetY, canvasHeight - scaleFactor * (maxY - minY), 0),
            });
        };

        const handleMouseUp = () => {
            window.removeEventListener("mousemove", handleMouseMove);
            window.removeEventListener("mouseup", handleMouseUp);
        };

        window.addEventListener("mousemove", handleMouseMove);
        window.addEventListener("mouseup", handleMouseUp);
    };

    return (
        <div className="flex flex-col items-start  gap-y-3 ">

            <canvas
                ref={canvasRef}
                width={750}
                height={400}
                style={{ display: 'block', cursor: 'pointer' }}
                onWheel={handleWheel}
                onMouseDown={handleMouseDown}
            ></canvas>

            <section className="flex items-start gap-x-12 w-full">
                <div className="overflow-y-auto border-2 flex flex-col gap-y-2 border-[#5d5d5d] rounded-lg p-3 w-[20vw] h-[25vh]">
                    <ul className="overflow-x-hidden overflow-y-auto">
                        {data.map((_, index) => (
                            <li
                                key={index}
                                className={`font-bold cursor-pointer w-max px-4 rounded-lg transition-all hover:bg-orange-400 text-lg ${index === generationIndex ? "bg-blue-500 " : ""
                                    }`}
                                onClick={() => setGenerationIndex(index)}
                            >
                                Iteration {index + 1}
                            </li>
                        ))}
                    </ul>
                </div>

                <div className="flex items-center gap-x-6">
                    <label className='text-lg font-semibold'>Animation Speed:</label>
                    <input
                        type="range"
                        min="1"
                        max="10"
                        value={speed}
                        onChange={(e) => setSpeed(parseInt(e.target.value, 10))}
                    />
                </div>
            </section>

            {/* <div className='overflow-y-auto border-2 flex flex-col gap-y-2 border-[#5d5d5d] rounded-lg px-4 py-2 w-full h-[20vh]'>
                <ul className='overflow-x-hidden overflow-y-auto'>
                    {data[generationIndex].particles.map((particle, idx) => (
                        <li key={idx} className='font-bold text-lg hover:text-orange-400'>
                            Particle {idx + 1}: Fitness {particle.fitness.toFixed(2)}
                        </li>
                    ))}
                </ul>
            </div> */}



        </div>
    );
}

export default ParticleVisualization;
