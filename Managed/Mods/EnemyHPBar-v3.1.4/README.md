# Enemy-HP-Bar
A Hollow Knight Mod add hpbar to enemy(for1.5.78)，need Satchel
## Install
1. Install the mod
2. Have `.png` image files
3. Run the game once
    1. This generates a folder in Location where EnemyHPBar installed named `CustomHPBar`,and will generate a `Default` folder.
4. Exit the game
5. Go to the `CustomHPBar` folder,and create a folder by yourself.(Yeah, like Custom Knight)
6. Move the image files here, `bg|fg|mg|ol|bossol|bossfg|bossbg.png`
7. Start the game
8. You can change skins in mod menu like ck.
## About CustomKnight
**If you install CustomKnight**, you can move all images to `Mods/CustomKnight/Skins/<skin>/HPBar`directory,
and when you change skin ,you will use HPBar in this skin first

## Anim Support
Now EnemHPBar support simple anim(Thanks Satchel Animation Code!),If you want to make hpbar anim,you should rename the images you use as `bossol_0.png`, `bossol_1.png`... and so on.
If you do this, the mod will generate `json` file like `bossol.json` in hpbar skin folder,the json format:
```json
{"fps":10.0,"loop":true}
``` 
you can edit anim fps,and whether it loop
