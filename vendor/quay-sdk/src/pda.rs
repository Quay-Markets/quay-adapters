//! PDA seed schemes — mirrors `onchain/program/src/pda.rs`.
//!
//! Every helper returns `(Pubkey, bump)` from `find_program_address`.

use solana_program::pubkey::Pubkey;

pub const SEED_GLOBAL: &[u8] = b"global";
pub const SEED_STRATEGY: &[u8] = b"strategy";
pub const SEED_MM: &[u8] = b"mm";
pub const SEED_VAULT: &[u8] = b"vault";

/// SPL Token program id (`TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA`).
pub const SPL_TOKEN_PROGRAM_ID: Pubkey = Pubkey::new_from_array([
    0x06, 0xdd, 0xf6, 0xe1, 0xd7, 0x65, 0xa1, 0x93, 0xd9, 0xcb, 0xe1, 0x46, 0xce, 0xeb, 0x79, 0xac,
    0x1c, 0xb4, 0x85, 0xed, 0x5f, 0x5b, 0x37, 0x91, 0x3a, 0x8c, 0xf5, 0x85, 0x7e, 0xff, 0x00, 0xa9,
]);

/// Token-2022 program id (`TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb`).
pub const TOKEN_2022_PROGRAM_ID: Pubkey = Pubkey::new_from_array([
    0x06, 0xdd, 0xf6, 0xe1, 0xee, 0x75, 0x8f, 0xde, 0x18, 0x42, 0x5d, 0xbc, 0xe4, 0x6c, 0xcd, 0xda,
    0xb6, 0x1a, 0xfc, 0x4d, 0x83, 0xb9, 0x0d, 0x27, 0xfe, 0xbd, 0xf9, 0x28, 0xd8, 0xa1, 0x8b, 0xfc,
]);

/// SPL Associated Token Account program id
/// (`ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL`).
pub const SPL_ATA_PROGRAM_ID: Pubkey = Pubkey::new_from_array([
    0x8c, 0x97, 0x25, 0x8f, 0x4e, 0x24, 0x89, 0xf1, 0xbb, 0x3d, 0x10, 0x29, 0x14, 0x8e, 0x0d, 0x83,
    0x0b, 0x5a, 0x13, 0x99, 0xda, 0xff, 0x10, 0x84, 0x04, 0x8e, 0x7b, 0xd8, 0xdb, 0xe9, 0xf8, 0x59,
]);

/// Is `program` one of the two supported token programs?
pub fn is_token_program(program: &Pubkey) -> bool {
    *program == SPL_TOKEN_PROGRAM_ID || *program == TOKEN_2022_PROGRAM_ID
}

/// Canonical `GlobalConfig` PDA — `["global"]`.
pub fn global_config_pda(program_id: &Pubkey) -> (Pubkey, u8) {
    Pubkey::find_program_address(&[SEED_GLOBAL], program_id)
}

/// Canonical `Strategy` PDA — `["strategy", owner, id]`.
pub fn strategy_pda(program_id: &Pubkey, owner: &Pubkey, id: u8) -> (Pubkey, u8) {
    Pubkey::find_program_address(&[SEED_STRATEGY, owner.as_ref(), &[id]], program_id)
}

/// Canonical `MarketMaker` PDA — `["mm", owner]`.
pub fn market_maker_pda(program_id: &Pubkey, owner: &Pubkey) -> (Pubkey, u8) {
    Pubkey::find_program_address(&[SEED_MM, owner.as_ref()], program_id)
}

/// Canonical vault token-account PDA — `["vault", market_maker, mint]`. The
/// vault's SPL authority is the `market_maker` PDA itself.
pub fn vault_pda(program_id: &Pubkey, market_maker: &Pubkey, mint: &Pubkey) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[SEED_VAULT, market_maker.as_ref(), mint.as_ref()],
        program_id,
    )
}

/// Canonical Associated Token Account address. ATA derivation includes the
/// token program id as a seed, so the same `(owner, mint)` pair has different
/// addresses under SPL Token vs Token-2022.
pub fn associated_token_account(
    owner: &Pubkey,
    mint: &Pubkey,
    token_program: &Pubkey,
) -> Pubkey {
    let (ata, _bump) = Pubkey::find_program_address(
        &[owner.as_ref(), token_program.as_ref(), mint.as_ref()],
        &SPL_ATA_PROGRAM_ID,
    );
    ata
}
