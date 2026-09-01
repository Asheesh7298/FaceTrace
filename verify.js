const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  const payloadHash = process.env.PAYLOAD_HASH;

  if (!payloadHash) {
    console.error("Error: PAYLOAD_HASH env var not set.");
    console.error("Usage: PAYLOAD_HASH=<hash> npx hardhat run scripts/verify.js --network localhost");
    process.exit(1);
  }

  // Load deployment info
  const deploymentFile = path.join(__dirname, `../../deployment_${network}.json`);
  if (!fs.existsSync(deploymentFile)) {
    console.error(`No deployment found for network: ${network}`);
    console.error(`Run deploy.js first.`);
    process.exit(1);
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentFile));
  console.log(`\nVerifying on: ${network}`);
  console.log(`Contract: ${deployment.contractAddress}`);
  console.log(`Payload hash: ${payloadHash}`);

  const faceProof = await hre.ethers.getContractAt(
    deployment.abi,
    deployment.contractAddress
  );

  const record = await faceProof.getProof(payloadHash);

  if (!record.payloadHash || record.payloadHash === "") {
    console.log("\n❌ NOT FOUND — hash not anchored on this network");
    process.exit(1);
  }

  console.log("\n✅ VERIFIED — Record found on-chain");
  console.log("─".repeat(50));
  console.log(`Face hash:        ${record.faceHash}`);
  console.log(`Content hash:     ${record.contentHash}`);
  console.log(`Source URL:       ${record.sourceUrl}`);
  console.log(`Confidence:       ${Number(record.confidenceScore) / 100}`);
  console.log(`Match found:      ${record.matchFound}`);
  console.log(
    `Anchored at:      ${new Date(Number(record.timestamp) * 1000).toISOString()}`
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
