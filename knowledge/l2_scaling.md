# Layer 2 Scaling

## Optimistic Rollups
Optimistic rollups (Arbitrum, Optimism, Base) post transaction data to Ethereum and assume transactions are valid by default. Fraud proofs allow anyone to challenge invalid state transitions within a 7-day window. This 7-day withdrawal period applies when bridging back to Ethereum directly — third-party bridges offer faster exits with liquidity risk. Gas fees are 10-50x lower than Ethereum L1. EVM-equivalent — existing Solidity contracts deploy without modification.

## Arbitrum
Arbitrum is the largest Ethereum L2 by TVL. Uses Nitro stack — EVM-compatible with WASM fraud proofs. Arbitrum One is the main chain. Arbitrum Nova uses AnyTrust — lower fees for gaming and social applications by trusting a data availability committee. Stylus enables Rust and C++ smart contracts alongside Solidity. ARB token governs the Arbitrum DAO.

## Optimism and the Superchain
Optimism introduced the OP Stack — an open-source rollup framework. Base (Coinbase), Zora, Mode, and dozens of other chains are built on the OP Stack, forming the Superchain — sharing sequencer infrastructure and bridging. The Superchain vision: hundreds of application-specific chains with shared security and interoperability. OP token governs the Optimism Collective using a bicameral governance model (Token House and Citizens House).

## Base
Base is Coinbase's L2 built on the OP Stack. No native token — ETH is used for gas. Backed by Coinbase's regulatory compliance focus. Onboarding from Coinbase directly to Base is seamless — key retail adoption driver. High activity in social applications (Farcaster ecosystem), meme coins, and creator monetization. Part of the Optimism Superchain.

## ZK Rollups
ZK rollups (zkSync Era, Starknet, Scroll, Polygon zkEVM) use cryptographic validity proofs — zero-knowledge proofs — to verify transaction correctness without a challenge period. Withdrawals are near-instant once proof is verified. Higher computational overhead for proof generation but stronger security guarantees than optimistic rollups. ZK-EVMs vary in EVM compatibility: Type 1 (full equivalence, slowest) to Type 4 (language-level, fastest).

## zkSync Era
zkSync Era by Matter Labs is a Type 4 ZK-EVM — compiles Solidity to zkSync's VM. Native account abstraction built in — enabling gas sponsorship and smart wallets. ZK token launched in 2024. Hyperchains framework allows app-specific ZK chains connected to Era. Strong developer tooling and ecosystem growth.

## Starknet
Starknet uses STARK proofs (no trusted setup, quantum-resistant). Cairo is the native programming language — not EVM-compatible, requiring contract rewrites. Higher performance ceiling than EVM-based ZK rollups. STRK token used for gas and governance. Starknet is particularly strong for gaming and applications requiring high computation throughput.

## Polygon
Polygon ecosystem: Polygon PoS (sidechain — not a true rollup), Polygon zkEVM (Type 2 ZK-EVM), Polygon CDK (chain development kit for ZK L2s). POL token replaced MATIC as the ecosystem token. AggLayer connects multiple ZK chains with unified liquidity and shared proofs — competing directly with the Optimism Superchain model. Polygon has strong enterprise adoption and gaming ecosystem.

## L2 Comparison
Transaction costs (approximate): Arbitrum One $0.02-0.10, Optimism $0.02-0.10, Base $0.01-0.05, zkSync Era $0.05-0.20, Starknet $0.05-0.30. Settlement finality: Optimistic rollups 7 days to L1, ZK rollups hours to L1 (proof generation time). EVM compatibility: Arbitrum/Base/Optimism full, zkSync partial, Starknet none (Cairo). TVL ranking (2024): Arbitrum, Base, Optimism, zkSync, Starknet.
