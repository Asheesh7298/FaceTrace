// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title FaceProof
 * @notice Anchors face identification pipeline results on-chain.
 *         Stores a JSON payload hash — face hash, content hash,
 *         source URL, confidence scores, and pipeline metadata.
 *         Emits an event per anchor so the full audit trail is
 *         permanently queryable from transaction logs.
 */
contract FaceProof {
    address public owner;

    struct ProofRecord {
        string  payloadHash;      // SHA-256 of the full JSON payload
        string  faceHash;         // SHA-256 of the input face crop
        string  contentHash;      // SHA-256 of matched content (or "none")
        string  sourceUrl;        // Best matched URL (or "none")
        uint256 confidenceScore;  // Overall confidence * 100 (e.g. 87 = 0.87)
        uint256 timestamp;        // Block timestamp
        bool    matchFound;       // Whether a social match was found
    }

    // payloadHash → ProofRecord
    mapping(string => ProofRecord) public proofs;

    // Full ordered list of anchored hashes (for audit trail)
    string[] public anchoredHashes;

    event Anchored(
        string indexed payloadHash,
        string faceHash,
        string sourceUrl,
        uint256 confidenceScore,
        bool matchFound,
        uint256 timestamp
    );

    event Verified(
        string indexed payloadHash,
        bool isValid,
        uint256 timestamp
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    /**
     * @notice Anchor a pipeline result on-chain.
     */
    function anchor(
        string memory _payloadHash,
        string memory _faceHash,
        string memory _contentHash,
        string memory _sourceUrl,
        uint256 _confidenceScore,
        bool _matchFound
    ) public onlyOwner {
        require(bytes(_payloadHash).length > 0, "Empty payload hash");
        require(
            bytes(proofs[_payloadHash].payloadHash).length == 0,
            "Already anchored"
        );

        proofs[_payloadHash] = ProofRecord({
            payloadHash:     _payloadHash,
            faceHash:        _faceHash,
            contentHash:     _contentHash,
            sourceUrl:       _sourceUrl,
            confidenceScore: _confidenceScore,
            timestamp:       block.timestamp,
            matchFound:      _matchFound
        });

        anchoredHashes.push(_payloadHash);

        emit Anchored(
            _payloadHash,
            _faceHash,
            _sourceUrl,
            _confidenceScore,
            _matchFound,
            block.timestamp
        );
    }

    /**
     * @notice Verify that a payload hash exists on-chain and return its record.
     */
    function verify(string memory _payloadHash)
        public
        returns (bool isValid, ProofRecord memory record)
    {
        record = proofs[_payloadHash];
        isValid = bytes(record.payloadHash).length > 0;

        emit Verified(_payloadHash, isValid, block.timestamp);
        return (isValid, record);
    }

    /**
     * @notice Get a proof record without emitting an event (read-only).
     */
    function getProof(string memory _payloadHash)
        public
        view
        returns (ProofRecord memory)
    {
        return proofs[_payloadHash];
    }

    /**
     * @notice Total number of anchored records.
     */
    function totalAnchored() public view returns (uint256) {
        return anchoredHashes.length;
    }
}
