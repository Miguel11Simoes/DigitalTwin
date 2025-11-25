import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
export default function Twin3D({
  vibration = 0.02, // g (máximo ~0.1 no simulador)
}) {
  const mountRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const modelRef = useRef(null);
  const basePosRef = useRef(new THREE.Vector3());
  const vibRef = useRef(vibration);
  const rafRef = useRef(0);
  useEffect(() => {
    vibRef.current = vibration;
  }, [vibration]);
  useEffect(() => {
    const mount = mountRef.current;
    const W = mount.clientWidth;
    const H = mount.clientHeight;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xC4C6CB);
    sceneRef.current = scene;
    const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 500);
    camera.position.set(7, 3.8, 10);
    cameraRef.current = camera;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    rendererRef.current = renderer;
    mount.appendChild(renderer.domElement);
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.8).texture;
    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.9);
    hemi.position.set(0, 6, 0);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(6, 10, 6);
    key.castShadow = true;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xffffff, 0.9);
    rim.position.set(-6, 6, -6);
    scene.add(rim);
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(60, 60),
      new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.9, metalness: 0.0 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.8;
    floor.receiveShadow = true;
    scene.add(floor);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enableZoom = true;
    controls.zoomSpeed = 1.2;
    controlsRef.current = controls;
    const loader = new GLTFLoader();
    loader.load(
      "/models/pump.glb",
      (gltf) => {
        const root = gltf.scene;
        root.traverse((o) => {
          if (o.isMesh) {
            o.castShadow = true;
            o.receiveShadow = true;
            if (o.material && o.material.color) {
              o.material.roughness = 0.6;
              o.material.metalness = 0.2;
            }
          }
        });
        root.scale.set(7, 7, 7);
        root.position.set(0, -0.25, 0);
        basePosRef.current.copy(root.position);
        modelRef.current = root;
        scene.add(root);
      },
      undefined,
      (e) => console.warn("Failed to load GLB", e)
    );
    let t = 0;
    const animate = () => {
      t += 0.016;
      const model = modelRef.current;
      if (model) {
        const v = Math.max(0, Math.min(0.1, vibRef.current || 0));
        const visibleAmp = Math.max(v, 0.002);
        const amp = visibleAmp * 0.50;
        model.position.x = basePosRef.current.x + Math.sin(t * 45) * amp;
        model.position.z = basePosRef.current.z + Math.cos(t * 41) * amp;
        model.position.y = basePosRef.current.y;
      }
      controls.update();
      renderer.render(scene, camera);
      rafRef.current = requestAnimationFrame(animate);
    };
    animate();
    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      cancelAnimationFrame(rafRef.current);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
      pmrem.dispose();
    };
  }, []);
  return <div ref={mountRef} style={{ width: "100%", height: "100%" }} />;
}

