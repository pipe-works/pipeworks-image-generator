# Example Prompt Files

This directory contains example prompt files for Pipeworks' file-based prompt builder system. These files demonstrate the structure and content format for generating game assets, particularly focused on a goblin fishing MUD (Multi-User Dungeon) theme.

## Quick Start

To use these example files:

```bash
# Copy all example files to your working inputs directory
cp -r examples/inputs/* src/inputs/

# Or create your own structure based on these examples
mkdir -p src/inputs
# Then copy specific categories you want
```

**Note:** The `src/inputs/` directory is gitignored by default, allowing you to customize prompts without affecting version control.

## Directory Structure

### `00_mud_assets/`
Game asset prompt files organized by category:

#### Equipment (`equipment/gob_fish_equip/`)
Fishing equipment organized by type and rarity:
- **Poles & Reels**: Basic fishing gear (`goblin_fishing_poles.txt`, `goblin_fishing_reels.txt`)
- **Hooks**: Organized by rarity (`hooks/`)
  - `goblin_hook_common.txt`
  - `goblin_hook_uncommon_sets.txt`
  - `goblin_hook_rare_sets.txt`
  - `goblin_hook_illegal_sets.txt`
  - `goblin_hook_forbidden_sets.txt`
- **Lures**: Same rarity structure as hooks (`lures/`)
- **Containers**: Cursed tackle boxes (`containers/`)
  - Rarity tiers: common, uncommon, rare, illegal, cursed, ABSOLUTELY_not
- **Shack Items**: Fishing shack decorations and tools

#### Wildlife (`wildlife/fish/`)
Fish species organized by rarity and special properties:
- `goblin_fish_species_common.txt`
- `goblin_fish_species_uncommon.txt`
- `goblin_fish_species_rare.txt`
- `goblin_fish_species_cursed.txt`
- `goblin_fish_species_talking.txt` - Special interactive fish
- `goblin_fish_species_absolutely_not.txt` - Extremely rare/dangerous

#### Clothing (`clothing/fishing/`)
Fishing-themed wearables:
- `patched_hats.txt`
- `fishing_belts.txt`
- `gloves.txt`

#### Books (`books/`)
- `fishing_books.txt` - In-game literature

#### Buildings (`buildings/hamlet/`)
- `build_hamlet.txt` - Settlement structures

### `01_styles/`
Art style modifiers for image generation, organized by media type:

Each style category has variants:
- `_p` - Primary style modifiers
- `_s` - Secondary style modifiers
- `_PNS` - Primary/Negative/Secondary combined

Available style categories:
- **Comic Book**: `z_comic_book_p.txt`, `z_comic_book_s.txt`
- **TV Styles**: `z_tv_styles_p.txt`, `z_tv_styles_s.txt`
- **Photo Styles**: `z_photo_styles_p.txt`, `z_photo_styles_s.txt`
- **Digital Art**: `z_digital_art_styles_p.txt`, `z_digital_art_styles_s.txt`, `z_digital_art_styles_PNS.txt`

### Root Files
- `art_style.txt` - Top-level art style definitions

## How Prompt Files Work

Each `.txt` file contains one prompt element per line. The prompt builder can:
- Select random lines from files
- Select specific line numbers or ranges
- Combine multiple files into segments (start, middle, end)
- Apply different selection modes per segment

### Example File Format
```
ancient fishing rod covered in barnacles
golden fishing pole with emerald inlays
simple wooden fishing stick
cursed rod that whispers at night
```

Each line represents a complete prompt fragment that can be randomly selected or specifically chosen during generation.

## Rarity Tier System

The example files use a progressive rarity system:
1. **Common** - Basic, everyday items
2. **Uncommon** - Slightly special or unusual
3. **Rare** - Valuable and unique
4. **Illegal** - Forbidden by in-game authorities
5. **Forbidden** - Dangerous or taboo items
6. **Cursed** - Magically afflicted
7. **ABSOLUTELY_not** - Extreme/joke tier (use with caution!)

This system allows for procedural generation across different power/rarity levels.

## Customization Tips

1. **Start small**: Copy only the categories you need
2. **Follow the naming convention**: Use descriptive names with underscores
3. **Organize by hierarchy**: Group related prompts in subdirectories
4. **One concept per line**: Keep each line focused on a single item/concept
5. **Test incrementally**: Add files gradually and test generation results

## Environment Configuration

The prompt builder's input directory is configured via environment variable:

```bash
# In your .env file
PIPEWORKS_INPUTS_DIR=src/inputs
```

See `.env.example` for all available configuration options.

## Need Help?

- See the main [README.md](../../README.md) for general Pipeworks usage
- Check [CLAUDE.md](../../CLAUDE.md) for development guidelines
- The prompt builder code is in `src/pipeworks/core/prompt_builder.py`
