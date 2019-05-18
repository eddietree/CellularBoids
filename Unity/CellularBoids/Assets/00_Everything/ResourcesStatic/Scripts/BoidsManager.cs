using System;
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
    const int NumGroups = 6;

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
        public float MaxRadius;
        public int HashGridSize;
    }
    public EnvironmentSettings envSettings = new EnvironmentSettings()
    {
        MaxRadius = 4f,
        HashGridSize = 64,
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

        public EnvironmentSettings envSettings;

        public void Execute(int i)
        {
            //var hash = (int)math.hash(new int3(math.floor(localToWorld.Position / cellRadius)));
            //hashMap.Add(hash, index);

            hashMap.Add(i, i);
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

        public EnvironmentSettings envSettings;
        public float deltaTime;

        public void Execute(int i)
        {
            // apply force from all
            float3 currVel = velocity[i];
            float3 currPos = position[i];
            int currGroupIndex = groupIndex[i];

            const float forceCoeffAttract = 0.2f;
            const float mass = 0.5f;
            float3 forceAccum = float3.zero;

            // TODO: optimize only the cells nearby
            for (int j = 0; j < NumCells; ++j)
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
            }

            // dist from edge
            float distFromEdge = Mathf.Max(0, envSettings.MaxRadius - math.length(currPos));
            forceAccum += -math.normalize(currPos) * Mathf.Exp(-5f * distFromEdge);

            // accel
            float3 accel = forceAccum / mass;
            currVel += accel * deltaTime;

            // drag
            const float drag = 3f;
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
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);

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
        _cellVelocities.Dispose();
        _cellGroupIndex.Dispose();
        _cellPositions.Dispose();
        _cellGroupsForceMatrix.Dispose();
        _cellTfmAccessArray.Dispose();

        if (hashMap.IsCreated)
            hashMap.Dispose();
    }

    void Setup()
    {
        _cellObjs = new GameObject[NumCells];
        _cellTfms = new Transform[NumCells];
        _cellRenderers = new MeshRenderer[NumCells];
        _cellMatProperyBlock = new MaterialPropertyBlock[NumCells];

        _cellGroupIndex = new NativeArray<int>(NumCells, Allocator.Persistent);
        _cellVelocities = new NativeArray<float3>(NumCells, Allocator.Persistent);
        _cellPositions = new NativeArray<float3>(NumCells, Allocator.Persistent);

        float radius = 2f;

        for( int i = 0; i < NumCells; ++i)
        {
            var newCellGroupIndex = UnityEngine.Random.Range(0, NumGroups);

            var newCell = GameObject.Instantiate(PfbCell);

            _cellPositions[i] = new float3( UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius));
            _cellObjs[i] = newCell;
            _cellTfms[i] = newCell.transform;
            _cellGroupIndex[i] = newCellGroupIndex;
            _cellVelocities[i] = (new float3(UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius)))* 0.1f;

            newCell.transform.localScale = Vector3.one * Mathf.Lerp(0.05f, 0.2f, Mathf.PerlinNoise((float)newCellGroupIndex / (float)NumGroups, 1f));

            // rendering
            var renderer = newCell.GetComponent<MeshRenderer>();
            MaterialPropertyBlock materialBlock;
            renderer.GetPropertyBlock(materialBlock = new MaterialPropertyBlock());
            _cellRenderers[i] = renderer;
            _cellMatProperyBlock[i] = materialBlock;

            var color = Color.HSVToRGB((float)newCellGroupIndex / (float)NumGroups, 1f, 1f);
            materialBlock.SetColor("_Color", color);
            renderer.SetPropertyBlock(materialBlock);


            var lineRenderer = renderer.GetComponentInChildren<TrailRenderer>();
            if (lineRenderer != null)
                lineRenderer.startColor = lineRenderer.endColor = color;
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
