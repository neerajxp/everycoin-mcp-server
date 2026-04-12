# Crypto Security

## Rug Pull Detection
Rug pull red flags: anonymous team with no doxxing or KYC, no audit from reputable firms (CertiK, Trail of Bits, OpenZeppelin), mint functions allowing unlimited token creation, unlocked or short-locked liquidity (less than 6 months), sell tax above 10%, honeypot contracts that prevent selling, single wallet holding more than 20% of supply. Tools: Token Sniffer, Honeypot.is, DEXTools, Bubble Maps for wallet concentration.

## Smart Contract Risks
Common vulnerabilities: reentrancy attacks (exploited in the DAO hack), integer overflow/underflow, flash loan attacks that manipulate oracle prices, access control flaws, front-running via MEV bots, logic errors in complex financial math. Always check: audit reports, bug bounty programs, time locks on admin functions, multi-sig on treasury, upgrade proxy risks.

## MEV and Sandwich Attacks
MEV (Maximal Extractable Value): bots monitor the mempool and reorder transactions for profit. Sandwich attacks buy before and sell after your DEX trade — inflating your purchase price and deflating your sell. Protection options: Flashbots Protect RPC endpoint, MEV Blocker, low slippage tolerance, using CoW Swap (batch auctions eliminate front-running), private mempool services.

## Wallet Security
Cold wallet best practices: purchase hardware wallets only from official manufacturers (Ledger, Trezor), never enter seed phrase digitally or photograph it, store seed on metal plate (Cryptosteel), use a strong passphrase as 25th word, verify firmware authenticity on setup. Hot wallet risks: browser extension vulnerabilities, clipboard hijacking, phishing sites. Hardware wallets should be used for holdings above $1,000.

## Bridge Security
Cross-chain bridges are the highest-risk DeFi infrastructure — over $2 billion lost to bridge hacks (Ronin, Wormhole, Nomad). Risk factors: centralized validator sets, upgrade keys without timelocks, complex cross-chain message verification. Safer alternatives: canonical bridges (Arbitrum Bridge, Optimism Gateway) using the rollup's own security. Third-party bridges carry smart contract and operational risk.

## Phishing and Social Engineering
Common attacks: fake token airdrop sites requesting wallet connection and signature, impersonator DMs on Discord and Telegram claiming to be support, malicious NFT metadata executing scripts, approval phishing — granting unlimited token spend to attacker contracts. Defense: never sign transactions you don't understand, use revoke.cash to audit and revoke approvals, treat unsolicited DMs as scams by default.

## Exchange Risk
Centralized exchange risks: insolvency (FTX collapse, 2022), withdrawal freezes, regulatory seizure. Best practices: not your keys not your coins, keep only trading amounts on CEX, use exchanges with proof of reserves, enable 2FA with hardware key not SMS, whitelist withdrawal addresses.

## Audit Firms and Standards
Reputable auditors: Trail of Bits, OpenZeppelin, Consensys Diligence, Halborn, Spearbit, Code4rena (competitive audits). A single audit is not a guarantee of safety — many audited protocols have been exploited. Look for multiple audits, an active bug bounty, and transparent incident response history.
