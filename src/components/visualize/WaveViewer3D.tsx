'use client';

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Float, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

interface WaveMeshProps {
    data: number[][];
    resolution: number;
}

const WaveSurface: React.FC<WaveMeshProps> = ({ data, resolution }) => {
    const meshRef = useRef<THREE.Mesh>(null);

    // Generate Geometry
    const geometry = useMemo(() => {
        const size = 10;
        return new THREE.PlaneGeometry(size, size, resolution - 1, resolution - 1);
    }, [resolution]);

    useFrame(() => {
        if (!meshRef.current || data.length === 0) return;

        const positions = meshRef.current.geometry.attributes.position.array as Float32Array;

        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const idx = (i * resolution + j) * 3;
                // height mapping
                const val = data[i][j];
                // Smoothly lerp height for animation
                const targetZ = (val - 0.5) * 3;
                positions[idx + 2] += (targetZ - positions[idx + 2]) * 0.1;
            }
        }
        meshRef.current.geometry.attributes.position.needsUpdate = true;
        meshRef.current.geometry.computeVertexNormals();
    });

    return (
        <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
            <meshStandardMaterial
                color="#3b82f6"
                wireframe
                transparent
                opacity={0.6}
                roughness={0.1}
                metalness={0.8}
                emissive="#1e3a8a"
                emissiveIntensity={0.5}
            />
        </mesh>
    );
};

const WaveViewer3D: React.FC<WaveMeshProps> = ({ data, resolution }) => {
    return (
        <div className="w-full aspect-square bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden relative shadow-2xl">
            <Canvas shadows>
                <PerspectiveCamera makeDefault position={[12, 12, 12]} fov={40} />
                <OrbitControls makeDefault enableDamping dampingFactor={0.05} />

                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1.5} color="#3b82f6" />
                <spotLight position={[-10, 20, 10]} angle={0.15} penumbra={1} intensity={2} color="#6366f1" />

                <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.5}>
                    <WaveSurface data={data} resolution={resolution} />
                </Float>

                <Environment preset="city" />
                <gridHelper args={[20, 20, 0x1e293b, 0x0f172a]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]} />
            </Canvas>

            <div className="absolute top-4 left-4 bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-[10px] text-slate-400 font-mono">
                3D WAVE INTERFACE
            </div>

            <div className="absolute top-4 right-4 flex gap-2">
                <div className="bg-blue-600/20 backdrop-blur-md px-3 py-1.5 rounded-lg border border-blue-500/30 text-[10px] text-blue-400 font-mono font-bold">
                    OPENGL RENDER
                </div>
            </div>

            <div className="absolute bottom-4 right-4 text-[8px] text-slate-600 font-mono">
                USE MOUSE TO ROTATE / ZOOM
            </div>
        </div>
    );
};

export default WaveViewer3D;
