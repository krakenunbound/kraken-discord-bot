# Kraken Discord Bot

A standalone ComfyUI custom node that turns your ComfyUI instance into a Discord image generation bot. No complex workflows needed - just one node that handles everything!

![Kraken Discord Bot](web/kraken_node_bg.png)

## Features

- **All-in-One Node**: Single node handles Discord bot, image generation, and response
- **Simple Setup**: Just add your Discord token, select a model, and queue
- **Discord Commands**: Users can generate images with `!generate <prompt>`
- **Style Presets**: Built-in styles (anime, photorealistic, fantasy, etc.)
- **Parameter Control**: Users can customize steps, CFG, size, seed via flags
- **Rate Limiting**: Configurable per-user cooldown to prevent spam
- **Queue Management**: Handles multiple requests with queue position feedback
- **Persistent Config**: Settings are saved and restored between sessions

## Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "Kraken Discord Bot"
3. Click Install

### Method 2: Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/kraken-discord-bot.git
cd kraken-discord-bot
pip install -r requirements.txt
```

### Method 3: Download ZIP
1. Download this repository as ZIP
2. Extract to `ComfyUI/custom_nodes/kraken-discord-bot`
3. Install requirements: `pip install discord.py requests`

## Setup

### 1. Create a Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name
3. Go to "Bot" section and click "Add Bot"
4. Under "Privileged Gateway Intents", enable **Message Content Intent**
5. Copy the bot token (you'll need this)

### 2. Invite Bot to Your Server

1. Go to "OAuth2" > "URL Generator"
2. Select scopes: `bot`
3. Select permissions: `Send Messages`, `Attach Files`, `Read Message History`
4. Copy the generated URL and open it to invite the bot

### 3. Configure the Node

1. Add the "Kraken Discord Bot (All-in-One)" node to your workflow
2. Paste your Discord token in the `discord_token` field
3. Select a checkpoint model
4. Configure default settings (steps, CFG, size)
5. Queue the workflow - the bot will start and wait for commands!

## Discord Commands

### Generate Images
```
!generate <prompt>
!gen <prompt>
!g <prompt>
```

### Optional Flags
```
--negative <text>     What to avoid in the image
--steps <1-100>       Number of generation steps (default: 20)
--cfg <1-30>          Guidance scale (default: 7)
--width <512-2048>    Image width (default: 1024)
--height <512-2048>   Image height (default: 1024)
--seed <number>       Specific seed for reproducibility
--style <name>        Style preset (see below)
```

### Size Presets
Use with `--width`:
- `square` (1024x1024)
- `landscape` (1216x832)
- `portrait` (832x1216)
- `wide` (1344x768)
- `tall` (768x1344)

### Style Presets
- `photorealistic` - Realistic photo style
- `anime` - Anime/manga style
- `fantasy` - Epic fantasy art
- `scifi` - Science fiction
- `artistic` - Painterly style
- `cinematic` - Movie-like shots
- `cute` - Adorable kawaii style
- `dark` - Dark and moody
- `vintage` - Retro aesthetic
- `minimalist` - Clean and simple

### Examples
```
!generate a majestic dragon flying over mountains
!gen cyberpunk city at night --style scifi --steps 25
!g portrait of a warrior --width portrait --style fantasy
!generate cute cat --negative ugly, blurry --seed 12345
```

### Other Commands
```
!help    - Show help message
!status  - Show bot status and queue
```

## Node Settings

| Setting | Description | Default |
|---------|-------------|---------|
| discord_token | Your Discord bot token | (required) |
| checkpoint | Model to use for generation | (required) |
| default_negative | Default negative prompt | ugly, blurry... |
| default_steps | Default generation steps | 20 |
| default_cfg | Default CFG scale | 7.0 |
| default_width | Default image width | 1024 |
| default_height | Default image height | 1024 |
| sampler_name | Sampler to use | euler_ancestral |
| scheduler | Scheduler to use | normal |
| denoise | Denoise strength | 1.0 |
| rate_limit_seconds | Cooldown between requests per user | 30 |
| max_queue_size | Maximum pending requests | 10 |
| allowed_channels | Restrict to specific channel IDs | (all) |

## Configuration File

Settings are automatically saved to:
```
kraken-discord-bot/kraken_discord/kraken_discord_config.json
```

Your Discord token and all settings persist between restarts.

## Troubleshooting

### Bot not responding
- Make sure "Message Content Intent" is enabled in Discord Developer Portal
- Check that the bot has permissions in the channel
- Verify the token is correct

### Generation errors
- Ensure you have a valid checkpoint selected
- Check ComfyUI console for error messages
- Make sure you have enough VRAM

### Rate limiting
- Users must wait 30 seconds (default) between requests
- Adjust `rate_limit_seconds` to change this

## Requirements

- ComfyUI (latest version recommended)
- Python 3.10+
- discord.py >= 2.0
- A Discord bot token
- GPU with sufficient VRAM for your chosen model

## License

MIT License - feel free to use, modify, and distribute!

## Credits

- Built for ComfyUI
- Uses discord.py for Discord integration
- Kraken theme styling included

---

Made with love for the AI art community!
