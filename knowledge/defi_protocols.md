# DeFi Protocols

## Uniswap
Uniswap V3 uses concentrated liquidity. Liquidity providers set custom price ranges, achieving up to 4000x capital efficiency compared to V2. Fees are earned only while price is within range. Higher fee tiers (1%) suit volatile pairs. Lower tiers (0.05%) suit stablecoins. Impermanent loss is amplified outside the range. V4 introduces hooks — custom logic on swaps and liquidity events.

## Aave
Aave is a decentralized lending protocol. Users supply collateral and borrow against it. Health Factor below 1.0 triggers liquidation — liquidators repay debt and claim collateral at a discount. Flash loans allow uncollateralized borrowing within a single transaction — used for arbitrage, collateral swaps, and liquidations. Aave V3 introduced efficiency mode (eMode) for correlated assets and isolation mode for new assets.

## Curve Finance
Curve specializes in stablecoin and pegged asset swaps using the StableSwap invariant — minimizing slippage for assets that should trade near 1:1. CRV token emissions incentivize liquidity. veCRV (vote-escrowed CRV) governs gauge weights — determining which pools receive emissions. The Curve Wars refer to protocols competing to accumulate veCRV voting power to direct rewards to their pools.

## GMX
GMX is a perpetuals and spot exchange on Arbitrum and Avalanche. Uses a multi-asset liquidity pool (GLP) that acts as the counterparty to traders. GLP holders earn fees but bear trader profit/loss risk. Zero price impact trades use Chainlink oracles for pricing. V2 introduces isolated markets and synthetic assets.

## Pendle Finance
Pendle separates yield-bearing tokens into principal tokens (PT) and yield tokens (YT). PT trades at a discount and redeems at face value at maturity — fixed yield. YT captures all variable yield until maturity — leveraged yield exposure. Useful for fixed-rate lending and yield speculation. TVL surged with LRT (liquid restaking token) yield markets.

## Lido
Lido is the largest liquid staking protocol. Users deposit ETH and receive stETH — a rebasing token representing staked ETH plus accruing rewards. stETH is widely used as DeFi collateral. Lido controls over 30% of all staked ETH — raising centralization concerns. wstETH (wrapped stETH) is preferred for DeFi integrations as it is non-rebasing.

## MakerDAO / Sky
MakerDAO issues DAI — a decentralized stablecoin collateralized by crypto assets. Users open Vaults, deposit collateral, and mint DAI against it. Liquidation occurs when collateral ratio falls below threshold. DAI Savings Rate (DSR) offers yield to DAI holders. MakerDAO rebranded to Sky with new tokens SKY and USDS.

## Compound
Compound is an algorithmic money market. Interest rates adjust automatically based on supply and demand. cTokens represent supplied assets and accrue interest. COMP token governs the protocol. Compound V3 (Comet) uses a single base asset model — more capital efficient and lower risk than V2.
