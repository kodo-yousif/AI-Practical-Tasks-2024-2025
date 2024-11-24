import { useRef, useEffect, useState } from 'react';

function ParticleVisualization({ data, goal }) {
    const canvasRef = useRef(null);
    const [generationIndex, setGenerationIndex] = useState(0);
    const [speed, setSpeed] = useState(1);
    const canvasWidth = 700;
    const canvasHeight = 500;
    const {goal_x, goal_y} = goal

    // Calculate bounds for all particles
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

    const calculateInitialZoom = () => {
        const { minX, minY, maxX, maxY } = calculateBounds();
        const contentWidth = (maxX - minX) * 10;  
        const contentHeight = (maxY - minY) * 10;

        // Calculate zoom to fit the content with some padding
        const horizontalZoom = (canvasWidth * 0.8) / contentWidth;
        const verticalZoom = (canvasHeight * 0.8) / contentHeight;

        return Math.min(horizontalZoom, verticalZoom, 1); 
    };

    const [zoom, setZoom] = useState(() => calculateInitialZoom());
    const [offset, setOffset] = useState(() => {
        const bounds = calculateBounds();
        const centerX = ((bounds.maxX + bounds.minX) / 2) * 10;
        const centerY = ((bounds.maxY + bounds.minY) / 2) * 10;
        return {
            x: canvasWidth / 2 - centerX,
            y: canvasHeight / 2 - centerY
        };
    });

    const { minX, minY, maxX, maxY } = calculateBounds();

    const scaleFactor = Math.min(
        (canvasWidth * 0.8) / (maxX - minX),
        (canvasHeight * 0.8) / (maxY - minY)
    ) * zoom;

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        const drawParticles = () => {
            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            ctx.translate(offset.x, offset.y);
            ctx.scale(zoom, zoom);

            const generationData = data[generationIndex];
            let { global_best, particles } = generationData;

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

            ctx.fillStyle = "orange";
            ctx.beginPath();
            ctx.arc(
                goal_x * 10 + canvas.width / 2,
                goal_y * 10 + canvas.height / 2,
                25,
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

    const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

    const handleWheel = (e) => {
        e.preventDefault();
        const zoomDelta = e.deltaY > 0 ? -0.1 : 0.1;
        const newZoom = clamp(zoom + zoomDelta, 0.1, 5); // Lowered minimum zoom to 0.1 for more zoom out capability
        setZoom(newZoom);
    };

    const handleMouseDown = (e) => {
        const rect = canvasRef.current.getBoundingClientRect();

        // Starting position based on mouse click
        const startX = e.clientX - rect.left - offset.x;
        const startY = e.clientY - rect.top - offset.y;

        const handleMouseMove = (moveEvent) => {
            // Calculate the new offset based on movement
            const newOffsetX = moveEvent.clientX - rect.left - startX;
            const newOffsetY = moveEvent.clientY - rect.top - startY;

            // Update the offset, clamped to prevent dragging out of bounds
            setOffset((prevOffset) => ({
                x: clamp(
                    newOffsetX,
                    canvasWidth - scaleFactor * (maxX - minX),
                    0
                ),
                y: clamp(
                    newOffsetY,
                    canvasHeight - scaleFactor * (maxY - minY),
                    0
                ),
            }));
        };

        const handleMouseUp = () => {
            window.removeEventListener("mousemove", handleMouseMove);
            window.removeEventListener("mouseup", handleMouseUp);
        };

        window.addEventListener("mousemove", handleMouseMove);
        window.addEventListener("mouseup", handleMouseUp);
    };


    return (
        <div className="flex flex-col items-start gap-y-3">
            <canvas
                ref={canvasRef}
                width={750}
                height={500}
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
        </div>
    );
}

export default ParticleVisualization;