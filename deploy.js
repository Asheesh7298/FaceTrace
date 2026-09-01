const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  console.log(`\nDeploying FaceProof to: ${network}`);

  const [deployer] = await hre.ethers.getSigners();
  console.log(`Deployer address: ${deployer.address}`);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log(`Deployer balance: ${hre.ethers.formatEther(balance)} ETH`);

  // Deploy
  const FaceProof = await hre.ethers.getContractFactory("FaceProof");
  const faceProof = await FaceProof.deploy();
  await faceProof.waitForDeployment();

  const contractAddress = await faceProof.getAddress();
  console.log(`\nFaceProof deployed at: ${contractAddress}`);

  if (network === "sepolia") {
    console.log(
      `Etherscan: https://sepolia.etherscan.io/address/${contractAddress}`
    );
  }

  // Save deployment info so Python pipeline can read it
  const deploymentInfo = {
    network,
    contractAddress,
    deployerAddress: deployer.address,
    deployedAt: new Date().toISOString(),
    abi: JSON.parse(
      fs.readFileSync(
        path.join(
          __dirname,
          "../artifacts/blockchain/contracts/FaceProof.sol/FaceProof.json"
        )
      )
    ).abi,
  };

  const outDir = path.join(__dirname, "../../");
  const outFile = path.join(outDir, `deployment_${network}.json`);
  fs.writeFileSync(outFile, JSON.stringify(deploymentInfo, null, 2));
  console.log(`\nDeployment info saved to: deployment_${network}.json`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
