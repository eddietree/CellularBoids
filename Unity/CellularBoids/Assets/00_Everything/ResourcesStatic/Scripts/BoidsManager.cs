﻿using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine.Jobs;
using Unity.Mathematics;
using Unity.Burst;
using UnityEngine.SceneManagement;

public class BoidsManager : MonoBehaviour
{
    const int NumCells = 2048;
    const int NumGroups = 12;

    public GameObject PfbCell; // prototype

    TransformAccessArray _cellTfmAccessArray;

    // each cell!
    Transform [] _cellTfms;
    GameObject [] _cellObjs;
    MeshRenderer[] _cellRenderers;
    MaterialPropertyBlock[] _cellMatProperyBlock;
    NativeArray<int> _cellGroupIndex;
    NativeArray<float3> _cellPositions;
    NativeArray<float3> _cellVelocities;

    public NativeMultiHashMap<int, int> hashMap;

    // grid stuff
    NativeArray<float> _cellGroupsForceMatrix;

    [Serializable]
    public struct EnvironmentSettings
    {
        [Range(0.01f, 1f)] public float Mass;
        [Range(1f, 10f)] public float Drag;
        [Range(1f, 10f)] public float MaxRadius;
        [HideInInspector][NonSerialized] public int HashGridCount;
    }
    public EnvironmentSettings envSettings = new EnvironmentSettings()
    {
        Mass = 0.5f,
        Drag = 3f,
        MaxRadius = 4f,
        HashGridCount = 8,
    };

    PositionUpdateJob _jobPos;
    CellularForceJob _jobCellularForce;
    HashCellsJob _jobHashMap;

    JobHandle _jobHandlePosition;
    JobHandle _jobsHandleCellularForce;
    JobHandle _jobsHandleHashMap;

    [BurstCompile]
    struct HashCellsJob : IJobParallelFor
    {
        public NativeMultiHashMap<int, int>.Concurrent hashMap;
        [ReadOnly] public NativeArray<float3> position;
        [ReadOnly] public EnvironmentSettings envSettings;

        public void Execute(int i)
        {
            float3 pos = position[i];
            float3 posNormalized = (pos / envSettings.MaxRadius)*0.5f + (new float3(0.5f,0.5f,0.5f)); // from [0,1]

            int3 gridIndex = new int3(
                math.clamp((int)(posNormalized.x * envSettings.HashGridCount),0, envSettings.HashGridCount-1),
                math.clamp((int)(posNormalized.y * envSettings.HashGridCount),0, envSettings.HashGridCount-1),
                math.clamp((int)(posNormalized.z * envSettings.HashGridCount),0, envSettings.HashGridCount-1)
            );

            var hash = (int)math.hash(gridIndex);
            hashMap.Add(hash, i);
            //var hash = (int)math.hash(new int3(math.floor(localToWorld.Position / cellRadius)));
            //hashMap.Add(hash, index);
        }
    }

    [BurstCompile]
    struct CellularForceJob : IJobParallelFor
    {
        // cells
        [ReadOnly] public NativeArray<float3> position;
        public NativeArray<float3> velocity; 
        [ReadOnly] public NativeArray<int> groupIndex;

        // global
        [ReadOnly] public NativeArray<float> forceMatrix;
        [ReadOnly] public int numCells;
        [ReadOnly] public int numGroups;

        [ReadOnly] public EnvironmentSettings envSettings;
        [ReadOnly] public NativeMultiHashMap<int, int> hashMap;
        [ReadOnly] public float deltaTime;

        public void Execute(int i)
        {
            // apply force from all
            float3 currVel = velocity[i];
            float3 currPos = position[i];
            int currGroupIndex = groupIndex[i];

            const float forceCoeffAttract = 0.2f;
            float3 forceAccum = float3.zero;

            float3 currPosNormalized = (currPos / envSettings.MaxRadius) * 0.5f + (new float3(0.5f, 0.5f, 0.5f)); // from [0,1]
            int3 currGridIndex = new int3(
                math.clamp((int)(currPosNormalized.x * envSettings.HashGridCount), 0, envSettings.HashGridCount - 1),
                math.clamp((int)(currPosNormalized.y * envSettings.HashGridCount), 0, envSettings.HashGridCount - 1),
                math.clamp((int)(currPosNormalized.z * envSettings.HashGridCount), 0, envSettings.HashGridCount - 1)
            );

            int3 currGridIndexMin = math.clamp(currGridIndex - new int3(1, 1, 1), new int3(0, 0, 0), new int3(envSettings.HashGridCount - 1, envSettings.HashGridCount - 1, envSettings.HashGridCount - 1));
            int3 currGridIndexMax = math.clamp(currGridIndex + new int3(1, 1, 1), new int3(0, 0, 0), new int3(envSettings.HashGridCount - 1, envSettings.HashGridCount - 1, envSettings.HashGridCount - 1));

            for(int gridX = currGridIndexMin.x; gridX <= currGridIndexMax.x; ++gridX)
            {
                for (int gridY = currGridIndexMin.y; gridY <= currGridIndexMax.y; ++gridY)
                {
                    for (int gridZ = currGridIndexMin.z; gridZ <= currGridIndexMax.z; ++gridZ)
                    {
                        int3 gridIndex = new int3(gridX, gridY, gridZ);
                        var gridHash = (int)math.hash(gridIndex);

                        bool found = hashMap.TryGetFirstValue(gridHash, out int otherIndex, out NativeMultiHashMapIterator<int> it);
                        while (found)
                        {
                            if (i != otherIndex)
                            {
                                float3 otherPos = position[otherIndex];
                                int otherGroupIndex = groupIndex[otherIndex];

                                float3 dirToOtherPos = otherPos - currPos;
                                float3 dirToOtherPosNorm = math.normalize(dirToOtherPos);
                                float distToOtherPosSqr = math.dot(dirToOtherPos, dirToOtherPos);

                                // repulsion
                                forceAccum -= dirToOtherPosNorm * Mathf.Exp(-35f * distToOtherPosSqr) * 4f;

                                // cellular attraction?
                                float cellularForce = forceMatrix[currGroupIndex * numGroups + otherGroupIndex];
                                forceAccum += cellularForce * dirToOtherPosNorm * Mathf.Exp(-1f * distToOtherPosSqr) * forceCoeffAttract;
                            }

                            // next item
                            found = hashMap.TryGetNextValue(out otherIndex, ref it);
                        }
                    }
                }
            }

            /*for (int j = 0; j < NumCells; ++j)
            {
                int otherIndex = j;

                if (i == otherIndex)
                    continue;

                float3 otherPos = position[otherIndex];
                int otherGroupIndex = groupIndex[otherIndex];

                float3 dirToOtherPos = otherPos - currPos;
                float3 dirToOtherPosNorm = math.normalize(dirToOtherPos);
                float distToOtherPosSqr = math.dot(dirToOtherPos, dirToOtherPos);

                // repulsion
                forceAccum -= dirToOtherPosNorm * Mathf.Exp(-35f * distToOtherPosSqr) * 4f;

                // cellular attraction?
                float cellularForce = forceMatrix[currGroupIndex * numGroups + otherGroupIndex];
                forceAccum += cellularForce * dirToOtherPosNorm * Mathf.Exp(-1f * distToOtherPosSqr) * forceCoeffAttract;
            }*/

            // dist from edge
            float distFromEdge = Mathf.Max(0, envSettings.MaxRadius - math.length(currPos));
            forceAccum += -math.normalize(currPos) * Mathf.Exp(-5f * distFromEdge);

            float3 dirToCenter = -math.normalize(currPos);
            float3 dirForce = math.cross(dirToCenter, new float3(0f, 1f, 0f));
            forceAccum += dirForce*0.4f;

            // accel
            float3 accel = forceAccum / envSettings.Mass;
            currVel += accel * deltaTime;

            // drag
            float drag = envSettings.Drag;
            currVel = currVel * (1f - deltaTime * drag);

            velocity[i] = currVel;
        }
    }

    [BurstCompile]
    struct PositionUpdateJob : IJobParallelForTransform
    {
        public NativeArray<float3> position;  // the velocities from AccelerationJob
        [ReadOnly] public NativeArray<float3> velocity;  // the velocities from AccelerationJob

        public float deltaTime;

        public void Execute(int i, TransformAccess transform)
        {
            float3 vel = velocity[i];
            float3 pos = position[i] + vel * deltaTime;
            position[i] = pos;
            transform.position = pos;

            if (math.lengthsq(vel) > 0.01f)
                transform.rotation = Quaternion.LookRotation(vel);
        }
    }

    void Start()
    {
        Setup();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.F5))
            Setup();

        if (Input.GetKeyDown(KeyCode.F6))
            ResetPositionsOnly();

        if (hashMap.IsCreated)
            hashMap.Dispose();
        hashMap = new NativeMultiHashMap<int, int>(NumCells, Allocator.TempJob);

        _jobHashMap = new HashCellsJob()
        {
            hashMap = hashMap.ToConcurrent(),
            position = _cellPositions,
            envSettings = envSettings
        };

        _jobCellularForce = new CellularForceJob()
        {
            numCells = NumCells,
            numGroups = NumGroups,
            position = _cellPositions,
            velocity = _cellVelocities,
            groupIndex = _cellGroupIndex,
            forceMatrix = _cellGroupsForceMatrix,

            hashMap = hashMap,
            envSettings = envSettings,
            deltaTime = Time.deltaTime,
        };

        _jobPos = new PositionUpdateJob()
        {
            position = _cellPositions,
            velocity = _cellVelocities,
            deltaTime = Time.deltaTime,
        };

        _jobsHandleHashMap = _jobHashMap.Schedule(NumCells, 128);
        _jobsHandleCellularForce = _jobCellularForce.Schedule(NumCells, 128, _jobsHandleHashMap);
        _jobHandlePosition = _jobPos.Schedule(_cellTfmAccessArray, _jobsHandleCellularForce);
    }

    public void LateUpdate()
    {
        _jobHandlePosition.Complete();
    }

    private void OnDestroy()
    {
        CleanUp();
    }

    void CleanUp()
    {
        if (_cellObjs != null)
        {
            for (int i = 0; i < _cellObjs.Length; ++i)
            {
                GameObject.Destroy(_cellObjs[i]);
            }

            _cellObjs = null;
        }

        if (_cellVelocities.IsCreated) _cellVelocities.Dispose();
        if (_cellGroupIndex.IsCreated) _cellGroupIndex.Dispose();
        if (_cellPositions.IsCreated) _cellPositions.Dispose();
        if (_cellGroupsForceMatrix.IsCreated) _cellGroupsForceMatrix.Dispose();
        if (_cellTfmAccessArray.isCreated) _cellTfmAccessArray.Dispose();
        if (hashMap.IsCreated) hashMap.Dispose();
    }

    void ResetPositionsOnly()
    {
        float radius = envSettings.MaxRadius * 0.5f;

        for (int i = 0; i < NumCells; ++i)
        {
            var newPos = new float3(UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius));
            _cellPositions[i] = _cellObjs[i].transform.position = newPos;
        }

        foreach( var trail in Transform.FindObjectsOfType<TrailRenderer>())
        {
            trail.Clear();
        }
    }

    void Setup()
    {
        CleanUp();

        _cellObjs = new GameObject[NumCells];
        _cellTfms = new Transform[NumCells];
        _cellRenderers = new MeshRenderer[NumCells];
        _cellMatProperyBlock = new MaterialPropertyBlock[NumCells];

        _cellGroupIndex = new NativeArray<int>(NumCells, Allocator.Persistent);
        _cellVelocities = new NativeArray<float3>(NumCells, Allocator.Persistent);
        _cellPositions = new NativeArray<float3>(NumCells, Allocator.Persistent);

        float radius = envSettings.MaxRadius * 0.5f;

        for( int i = 0; i < NumCells; ++i)
        {
            var newCellGroupIndex = UnityEngine.Random.Range(0, NumGroups);

            var newCell = GameObject.Instantiate(PfbCell);

            _cellPositions[i] = new float3( UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius));
            _cellObjs[i] = newCell;
            _cellTfms[i] = newCell.transform;
            _cellGroupIndex[i] = newCellGroupIndex;
            _cellVelocities[i] = (new float3(UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius)))* 0.1f;

            newCell.transform.localScale = Vector3.one * Mathf.Lerp(0.03f, 0.25f, Mathf.PerlinNoise((float)newCellGroupIndex / (float)NumGroups, 1f));

            // rendering
            var renderer = newCell.GetComponent<MeshRenderer>();
            MaterialPropertyBlock materialBlock;
            renderer.GetPropertyBlock(materialBlock = new MaterialPropertyBlock());
            _cellRenderers[i] = renderer;
            _cellMatProperyBlock[i] = materialBlock;

            var color = Color.HSVToRGB((float)newCellGroupIndex / (float)NumGroups, 0.75f, 1f);
            var emission = newCellGroupIndex == 0 ? 2f : 0f;// Mathf.Pow( Mathf.PerlinNoise((float)newCellGroupIndex / (float)NumGroups, 1f), 10f) * 50f;
            materialBlock.SetColor("_Color", color);
            materialBlock.SetColor("_EmissionColor", Color.white * emission);
            renderer.SetPropertyBlock(materialBlock);

            var lineRenderer = renderer.GetComponentInChildren<TrailRenderer>();
            if (lineRenderer != null)
            {
                lineRenderer.SetPropertyBlock(materialBlock);
                //lineRenderer.startColor = lineRenderer.endColor = color;
            }
        }

        // add random lights
        for(int i = 0; i < 3; ++i)
        {
            var cellObj = _cellObjs[UnityEngine.Random.Range(0, _cellObjs.Length)];

            var light = cellObj.AddComponent<Light>();
            light.type = LightType.Point;
            light.range = 1.5f;
            light.intensity = 15f;
        }

        _cellTfmAccessArray = new TransformAccessArray(_cellTfms);

        // force matrix
        _cellGroupsForceMatrix = new NativeArray<float>(NumGroups * NumGroups, Allocator.Persistent);
        for(int i = 0; i < _cellGroupsForceMatrix.Length; ++i)
        {
            _cellGroupsForceMatrix[i] = UnityEngine.Random.Range(-1f, 1f);
        }

    }
}
