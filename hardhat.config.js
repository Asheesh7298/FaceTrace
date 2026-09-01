require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

module.exports = {
  solidity: "0.8.24",
  networks: {
    // Local Hardhat node — always works, no keys needed
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 31337,
    },
    // Sepolia testnet — needs Infura RPC + wallet private key
    sepolia: {
      url: process.env.INFURA_SEPOLIA_URL || "",
      accounts: process.env.DEPLOYER_PRIVATE_KEY
        ? [process.env.DEPLOYER_PRIVATE_KEY]
        : [],
      chainId: 11155111,
    },
  },
  paths: {
    sources: "./blockchain/contracts",
    scripts: "./blockchain/scripts",
    artifacts: "./blockchain/artifacts",
  },
};
