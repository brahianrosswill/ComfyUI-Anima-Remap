# ComfyUI-Anima-Remap

[English](README.md) | [日本語](README_ja.md)

> **補足（Anima-3.8B v1.1以降について）：** Animaのチェックポイントによっては、Qwen3.5 4B用のcross-attention部分（`semantic_attentions.*`などのconnector関連キー）が、別ファイルのアダプターとしてではなく、DiT本体と同一ファイルに同梱されている場合があります（Anima-3.8B v1.1の"Semantic Connector v2"以降でこの形になりました）。どちらの形で配布されていても、この部分がremap処理の対象になることはありません——検出・remapともに`net.blocks.N`というキーのみを見ており、connector側のキーはこのパターンに一致しないためです。実機確認済み：Qwen3.5を接続しない状態でも、ネイティブのQwen3 0.6B経路のみで生成は問題なく行えます。詳細は下記「Anima-3.8B（52層）対応について」を参照してください。

Anima系モデル(従来Anima/28層、Anima-2.9B/40層、Anima-3.8B/52層、そして今後登場するかもしれない世代)を横断して、LoRA適用・モデルマージ時のレイヤー構造の違いを自動的に吸収するComfyUIカスタムノードパッケージです。どの2世代の組み合わせであっても対応します。Anima専用のランダムLoRAローダー(フォルダ指定、同じ自動リマップ機構を内蔵)も含まれます。

このリポジトリは、更新を停止した[ComfyUI-Anima29B-Remap](https://github.com/shin131002/ComfyUI-Anima29B-Remap)の後継です。ノード内部IDは変更していないので、そちらのリポジトリ向けに作ったワークフローもそのまま読み込めます。

> ⚠️ **このリポジトリと旧`ComfyUI-Anima29B-Remap`を同時にインストールしないでください。** 両パッケージは同じノードIDを登録するため、両方入れているとノード登録が衝突します。移行する場合は、先に旧リポジトリを削除してください。

## これは何のためのツールか

これまでの各Anima世代は、前の世代の各層の間に新規層を挿入する形で拡張されてきました(Animaの28層→Anima-2.9Bの40層→Anima-3.8Bの52層)。このため、旧世代用に作られたLoRAやマージ用モデルをそのまま新しい世代に適用すると、**層のインデックス(番号)がズレて全く別の層に誤って適用されてしまい**、崩れた画像(いわゆる「落書きレベル」の破綻)や、モデルマージ時のノイズ画像の原因になります。

本パッケージは、各層拡張時のインデックス対応情報を記録した`expand_manifest.json`群を使って、どの2世代の組み合わせであっても、このズレを自動検出・補正した上でLoRA適用やモデルマージを行います。

## フォルダ構成

```
ComfyUI-Anima-Remap/
├── LICENSE                                  # コードのライセンス(MIT)
├── README.md                                # 英語版README
├── README_ja.md                             # このファイル(日本語)
├── __init__.py                              # ノード登録
├── mapping/
│   ├── expand_manifest_28_40.json           # 公式のAnima→Anima-2.9Bマニフェスト
│   ├── expand_manifest_40_52.json           # Anima-2.9B→Anima-3.8B(52層)マニフェスト(復元データ、出典は下記参照)
│   ├── expand_manifest_28_52_composed.json  # Anima→Anima-3.8B、上記2つから自動合成
│   └── scripts/
│       └── compose_manifests.py             # メンテナンス用ツール: 隣接世代のmanifestから全ペアを再生成する。ノード実行時には使用しない
└── nodes/
    ├── __init__.py                          # (空、パッケージ化のため)
    ├── anima_common.py                      # 共通ロジック(層検出・manifest自動選択・マッピング計算)
    ├── lora_remap_anima.py                  # LoRAタグローダー(自動リマップ)
    ├── lora_remap_extended_anima.py         # LoRAタグローダー拡張版(実験的、前後ブレンド)
    ├── model_merge_anima.py                 # モデルマージ(自動リマップ)
    ├── model_merge_extended_anima.py        # モデルマージ拡張版(実験的、前後ブレンド)
    ├── anima_random_lora_loader.py          # ランダムLoRAローダー、3フォルダ版(自動リマップ)
    └── anima_filtered_random_lora_loader.py # ランダムLoRAローダー、1フォルダ+キーワードフィルタ版(自動リマップ)
```

## インストール

このフォルダごと`ComfyUI/custom_nodes/`直下に配置し、ComfyUIを再起動してください。

```
ComfyUI/custom_nodes/ComfyUI-Anima-Remap/
```

またはgit cloneでも配置できます。

```
cd ComfyUI/custom_nodes/
git clone https://github.com/shin131002/ComfyUI-Anima-Remap.git
```

> 旧`ComfyUI-Anima29B-Remap`を既に導入済みの場合は、先にそちらのフォルダを削除してください。両パッケージは同じノードIDを登録するため、両方入れているとノード登録が衝突します。

再起動後、ノード検索(ダブルクリック)で「Anima」と入力すると、以下のノードが見つかります。

- **Anima LoRA Tag Loader (Auto Remap)**
- **Anima Model Merge (Auto Remap)**
- **Anima LoRA Tag Loader Extended (Experimental)** — 実験的機能。詳細は後述
- **Anima Model Merge Extended (Experimental)** — 実験的機能。詳細は後述
- **Anima Random LoRA Loader (28/40/52 Auto)** — フォルダ指定のランダム選択、3グループ。詳細は後述
- **Anima Filtered Random LoRA Loader (28/40/52 Auto)** — フォルダ指定のランダム選択、1フォルダ+キーワードフィルタ。詳細は後述

---

## ノード1: Anima LoRA Tag Loader (Auto Remap)

`<lora:名前:重み>`というタグ構文をプロンプト文字列から解析してLoRAを適用する、LoRA Tag Power Loader系ノードと同じ使い勝手のローダーです。タグごとに、そのLoRA自身のブロック数と接続されたモデルのブロック数を検出し、両者が異なる場合は自動でキー名をリマップしてから適用します。

![ノード1: Anima LoRA Tag Loader (Auto Remap)](./images/01.jpg)

### 入力

| 名前 | 型 | 説明 |
|---|---|---|
| `model` | MODEL | LoRAを適用するモデル |
| `clip` | CLIP(任意) | LoRAを適用するCLIP |
| `text` | STRING | `<lora:名前:重み>`タグを含むプロンプト文字列 |
| `default_weight` | FLOAT | タグ内で重み省略時のデフォルト値 |
| `weight_multiplier` | FLOAT | 全LoRAの重みに一律で掛ける倍率 |
| `auto_remap` | BOOLEAN | ONで自動リマップ機構を有効化(デフォルトON) |
| `save_remapped` | BOOLEAN | ONで、その場でリマップを行った際に**変換後のLoRAをファイルとしてディスクに保存**する(デフォルトOFF、詳細・注意点は下記) |
| `extend_to_new_layers` | BOOLEAN | 【実験的機能】ONで、新規12層にもLoRAの効果を(近似的に)適用する(デフォルトOFF) |
| `extend_strength` | FLOAT | `extend_to_new_layers`使用時の、新規層への適用強度(タグの重みと掛け算で効く) |
| `manifest` | ドロップダウン | デフォルトの`Auto (Recommended)`は、そのLoRA自身のブロック数と接続モデルのブロック数から、タグごとに正しいmanifestを自動選択します。混在するプロンプト(28層LoRAと40層LoRAを両方参照し、接続モデルが52層、等)でも、それぞれ正しいmanifestが個別に適用されます。特定のファイルを選択すると、検出結果に関わらず全タグでそれを強制します |

### 出力

| 名前 | 型 | 説明 |
|---|---|---|
| `model` | MODEL | LoRA適用後のモデル |
| `clip` | CLIP | LoRA適用後のCLIP |
| `text` | STRING | LoRAタグを取り除いた後のプロンプト文字列(後段のCLIP Text Encoderへ) |

### タグ構文

```
1girl, masterpiece <lora:my_old_anima_style:0.8> outdoors
```

`<lora:名前:model重み:clip重み>`のように、clip側の重みを別指定することも可能です(省略時はmodel重みと同じ値)。

### 自動判定のロジック(概要)

各`<lora:名前:重み>`タグごとに、以下を個別に行います。

1. 接続された`model`の総ブロック数を検出(`net.blocks.N`キーの最大値+1、`llm_adapter.blocks`は判定対象から除外)
2. そのLoRAファイル自体が何ブロック分のキーを持つかを同様に検出
3. LoRAの方がモデルよりブロック数が少なければ、その(LoRAのブロック数→モデルのブロック数)の組み合わせに対応するmanifestを解決してリマップする(`manifest`項目参照)。LoRAが既に同等以上のブロック数を持つ場合はそのまま適用する(「上回っている」場合の扱いは下記参照)

タグごとに個別処理されるため、異なるAnima世代のLoRAが混在する1つのプロンプトを同じモデルに適用する場合でも、それぞれのLoRAが自分自身の世代に応じて正しく処理されます(ノード実行全体で1回だけ判定する、という方式ではありません)。

> ⚠️ **LoRAの方が接続モデルより多くの層を参照している場合**(例: Anima-3.8B用に作られたLoRAをAnima-2.9Bモデルに適用しようとした場合)、層数を減らす方向のリマップは不可能です(高いインデックスのテンソルの行き場がありません)。この場合、一致しないテンソルをComfyUIが黙ってスキップして中途半端な適用結果になるのではなく、**エラーを発生させて処理を停止します**。この判定は、キャッシュ利用・その場でのリマップ・そのまま適用のどの経路を通った場合でも、最終的なLoRAに対して行われます。

### リマップ結果のキャッシュ保存(`save_remapped`)について

`save_remapped`は、**リマップ変換後のLoRAをファイルとしてディスクに保存するかどうか**を切り替えるトグルです。保存しておくことで、2回目以降の実行ではリマップ処理そのものをスキップし、保存済みファイルをそのまま読み込むだけで済むようになります。

キャッシュのファイル名には**リマップ先のブロック数**が埋め込まれます(例: `<名前>_animaremap52.<拡張子>`)。LoRA名だけでなく変換先の世代も含めて区別されるため、40層モデル向けにキャッシュしたファイルを、後で52層モデルに接続した際に誤って再利用してしまうことはありません。(LoRA, 変換先世代)の組み合わせごとに別々のキャッシュファイルになります。

> ⚠️ **注意: `manifest`・`extend_strength`・`extend_to_new_layers`を調整しながら検討している間は、`save_remapped`はOFFのままにしておくことを強くおすすめします。**
>
> キャッシュファイルが一度でも保存されると、**その(LoRA, 変換先世代)の組み合わせについては、その後`manifest`・`extend_to_new_layers`・`extend_strength`の値をいくら変更しても、新しい設定は一切反映されません**。既にキャッシュが存在する限り、そのファイルの中身(=保存した瞬間の設定で焼き付けられた結果)がそのまま優先して読み込まれ続けるためです。
>
> そのため、おすすめの運用は次の通りです。
>
> 1. まず`save_remapped`は**OFF**のまま、`manifest`・`extend_to_new_layers`・`extend_strength`をいろいろ試しながら好みの設定を探す(この間は毎回その場でリマップされるだけで、ファイルには保存されません)
> 2. 設定が決まったら、その時だけ`save_remapped`を**ON**にして1回実行し、確定版のキャッシュファイルを書き出す
> 3. 設定を変えて試し直したくなったら、生成されたキャッシュファイルを一度削除してから、また1に戻る

あるタグについての動作の流れは以下の通りです(そのLoRAがリマップ対象と判定された場合のみ発生します)。

1. **まず`<元のLoRA名>_animaremap<変換先ブロック数>.<拡張子>`という名前のファイルが、元LoRAと同じ検索方法(同フォルダ/サブフォルダ含む)で既に存在しないか確認する**
2. **存在する場合**: そのファイルをそのまま読み込んで適用する。リマップ処理自体は一切行わない(=`save_remapped`の設定に関わらず、既存のキャッシュファイルがあれば常にそれが優先される)
3. **存在しない場合**: 元のLoRAを読み込み、その場でキーをリマップして適用する。この時点で`save_remapped`がONなら、リマップ結果を`<元のLoRA名>_animaremap<変換先ブロック数>.<拡張子>`として**元LoRAと同じフォルダに新規保存する**(既に他のタイミングで同名ファイルが作られていた場合は上書きせずスキップする)。`save_remapped`がOFFの場合は、適用はするがファイルへの保存は行わない(次回実行時もこの「その場でリマップ」処理を毎回繰り返すことになる)

つまり`save_remapped`をONにしておくと、初回実行時に自動生成されたキャッシュファイルが、同じ(LoRA, 変換先世代)の組み合わせについて以降ずっと使われ続けるため、**2回目以降はリマップ処理のオーバーヘッドなしで高速に適用できる**、という仕組みです。

### 注意点

- `llm_adapter.blocks`(6層、これまでの拡張の対象外)は、メインの`net.blocks`とは別構造として扱われ、常にリマップ対象から除外されます
- LoRAのキー命名は「ドット区切り(`net.blocks.N.`)」「kohya形式のアンダースコア区切り(`..._blocks_N_...`)」の2パターンに対応しています。それ以外の命名規則の場合は検出されず、リマップなしでそのまま適用されます
- ノード5・6(ランダムローダー)は`.safetensors`・`.pt`・`.ckpt`を対象にスキャンします

---

## ノード2: Anima Model Merge (Auto Remap)

2つのAnimaモデル(MODEL)を、レイヤー構造の違いを自動吸収しながらマージします。どの2世代の組み合わせでも対応します。

![ノード2: Anima Model Merge (Auto Remap)](./images/02.jpg)

### 入力

| 名前 | 型 | 説明 |
|---|---|---|
| `model_1` | MODEL | マージ元モデル1(上側スロット) |
| `model_2` | MODEL | マージ元モデル2(下側スロット) |
| `merge_ratio` | FLOAT(0.0〜1.0) | `model_1`側の混合比率 |
| `extend_ratio` | FLOAT(0.0〜1.0) | 【実験的機能】新規挿入層にも、コピー元の小さい世代側の層の値を(近似的に)ブレンドする度合い(デフォルト0.0 = 従来通り新規層は大きい世代側のまま) |
| `manifest` | ドロップダウン | デフォルトの`Auto (Recommended)`は、`model_1`/`model_2`の検出ブロック数からmanifestを自動解決します。特定のファイルを選択すると、それを強制します |

### 出力

| 名前 | 型 |
|---|---|
| `model` | MODEL |

### `merge_ratio`の意味

- `1.0` → 出力は`model_1`(上側)100%
- `0.0` → 出力は`model_2`(下側)100%
- `0.5` → 50:50でブレンド

### アーキテクチャに応じた自動切り替え

| 状況 | 出力 |
|---|---|
| 両方とも同じブロック数(同じ世代の派生同士) | そのブロック数のまま、直接マージ(リマップ不要) |
| ブロック数が異なる | **常に大きい方のブロック数で出力**。小さい世代側の重みをリマップした上で、共有部分のみ`merge_ratio`でブレンド。新規挿入層は、`model_1`・`model_2`どちらのスロットに繋がれていても**デフォルトでは常に大きい世代側モデルの値をそのまま使用**(`merge_ratio`の影響を受けない。`extend_ratio`で挙動を変更可能、詳細は下記) |
| ブロック数が異なり、かつその組み合わせに対応するmanifestが無い | リマップなしの直接マージにフォールバックし、警告をログ出力する(アーキテクチャが実際に異なる場合、結果が不正確になる可能性あり) |

### `extend_ratio`について(新規挿入層への拡張、実験的機能)

デフォルト(`extend_ratio = 0.0`)では、新規挿入層は常に大きい世代側モデルの値をそのまま使用し、`merge_ratio`や小さい世代側モデルの影響を一切受けません。

`extend_ratio`を0より大きくすると、解決されたmanifestの`inserted_to_source`(各新規層が初期化時にどの旧層からコピーされたか)を使い、新規層の値に小さい世代側モデルの対応する旧層の値を(リマップした上で)ブレンドします。

```
新規層の最終的な値 = (1 - extend_ratio) × 大きい世代側モデル自身の値 + extend_ratio × 小さい世代側モデルの対応する旧層の値
```

`extend_ratio = 1.0`にすると、新規層についても小さい世代側モデルの(近似的な)値で完全に置き換わります。LoRA側の`extend_to_new_layers`/`extend_strength`と同じ考え方に基づいていますが、こちらは独立した機能で、**「正解」は存在しない近似処理**である点は同様です。

なお、このノードには保存機能がないため(下記参照)、LoRA側の`save_remapped`のような「一度保存すると設定変更が反映されなくなる」という注意点はありません。マージ結果を保存する際は、その都度明示的に`ModelSave`ノードを実行する必要があるため、意図しない古い設定のファイルを使い続けてしまう心配は基本的にありません。

### バイパス時の挙動

ノードを「Bypass」モードにした場合、ComfyUI標準の挙動により、型が一致する最初の入力(`model_1`、上側)がそのまま出力されます。これは本ノード固有の実装ではなく、ComfyUI自体の標準機能によるものです。

### 保存について

このノード自体には保存機能はありません。マージ結果を保存する場合は、ComfyUI標準の**`ModelSave`**ノード(`advanced/model_merging`カテゴリ)を`model`出力に接続してください。

```
[Anima Model Merge] --MODEL--> [ModelSave]
```

`ModelSave`はUNet(拡散モデル)部分のみを保存します(CLIP/VAEは含まれません)。保存先はデフォルトで`ComfyUI/output`配下になるため、生成用に使う場合は`models/diffusion_models`等へ移動してください。

---

## ノード3・4: Extended (Experimental)版について

`Anima LoRA Tag Loader Extended (Experimental)`と`Anima Model Merge Extended (Experimental)`は、通常版のノード(ノード1・2)を一切変更せず、**別ファイル・別ノードとして追加した実験的なバリエーション**です。通常版はこれまで通りの動作のまま安心して使い続けられます。

![ノード3: Anima LoRA Tag Loader Extended (Experimental)](./images/03.jpg)

![ノード4: Anima Model Merge Extended (Experimental)](./images/04.jpg)

### 何が拡張されているか

通常版の`extend_to_new_layers`/`extend_strength`(LoRA)や`extend_ratio`(モデルマージ)は、新規挿入層への適用時、解決されたmanifestの`inserted_to_source`が示す**「直前の旧層」のみ**を情報源としていました。

Extended版では、これを**「直前(前側)」と「直後(後ろ側)」の両方の旧層を、指定した比率でブレンドする**形に一般化しています。

- **LoRA Extended**: `blend_ratio`(0.0〜1.0)を追加。前側の重みが`blend_ratio`、後ろ側の重みが`1 - blend_ratio`
- **Model Merge Extended**: 同じく`blend_ratio`を追加。既存の`extend_ratio`と組み合わせて使います(`extend_ratio`が「大きい世代側自身の値」と「小さい世代側由来のブレンド値」の混合比率、`blend_ratio`がその「小さい世代側由来のブレンド値」自体を前後どちらの層から作るかの比率)

```
新規層への適用値 = 前側(直前の旧層)の値 × blend_ratio + 後側(直後の旧層)の値 × (1 - blend_ratio)
```

**`blend_ratio = 1.0`のとき、後側の寄与が0になるため、通常版(ノード1・2)と数値まで完全に同じ結果になります**(テスト済み)。`blend_ratio`を下げていくと、徐々に「直後の層」の影響が混ざっていきます。

> **端のケースについて:** ブロック列の両端付近に挿入された層は、前後どちらか片方のneighborしか存在しない場合があります(最初の挿入層には「前側」が無く、最後の挿入層には「後側」が無い)。この場合`blend_ratio`は適用されず、**存在する側の唯一のneighborを常に全強度で使用**します。`blend_ratio`が存在しない側を優先する値になっていても、その層への拡張がゼロになってしまうことはありません。

### キャッシュファイルについて

LoRA Extended版のキャッシュファイルは、Extended版であることと変換先のブロック数の両方を示すサフィックス(例: `_animaremap52_ext`)で保存されるため、通常版のキャッシュ(`_animaremap52`)や、別の変換先向けのキャッシュと衝突することはありません。通常版と同様、`manifest`・`extend_to_new_layers`・`blend_ratio`・`extend_strength`を変更しながら検討する間は`save_remapped`をOFFのままにしておくことを推奨します(キャッシュは保存した瞬間の設定しか反映しない、という注意点は通常版と同じです)。

通常版・Extended版どちらのLoRAノードも、上記と同じ「タグごとのmanifest自動解決」と「層数不一致時は停止する」挙動を共有しています。接続モデルより多い層を参照するLoRAは縮小方向にリマップできないため、部分的に適用してしまうのではなくエラーを出して停止します。

### 位置づけ

あくまで実験目的の追加ノードです。「新規層への適用方法をもう少し細かく制御したい」と感じた時に使うもので、通常版だけで十分な場合は無理に使う必要はありません。

---

## ノード5・6: Anima Random LoRA Loader / Anima Filtered Random LoRA Loader (28/40/52 Auto)

[shin131002/RandomLoRALoader](https://github.com/shin131002/RandomLoRALoader)の「Random LoRA Loader」(3フォルダ同時選択)と「Filtered Random LoRA Loader」(1フォルダ+キーワードフィルタ)をAnima専用に移植したノードです。ノード1〜4と同じ自動リマップ機構をそのまま内蔵しており、ランダムに選ばれたLoRAごとに必要であれば自動でリマップしてから適用します。タグを手打ちする必要はありません。

![ノード5: Anima Random LoRA Loader (28/40/52 Auto)](./images/05.jpg)

![ノード6: Anima Filtered Random LoRA Loader (28/40/52 Auto)](./images/06.jpg)

**Anima Random LoRA Loader**は最大3フォルダ(例: スタイル/キャラクター/コンセプト)から同時に選択し、フォルダごとに強度範囲・選択数を設定できます。**Anima Filtered Random LoRA Loader**は単一フォルダ+キーワードフィルタ(AND/OR、フレーズ一致、メタデータ検索オプション)構成で、大規模で雑多なLoRAコレクションに向いています。

トリガーワード抽出・プレビュー画像出力・強度のランダム範囲指定(`"0.4-0.8"`等)は、元のRandomLoRALoaderパッケージと同じ仕様です。これらの詳細は元プロジェクトのREADMEを参照してください。

### 元のRandomLoRALoaderとの違い

- **Anima専用です。** SD1.5/SDXL/Flux等の汎用ローダーではありません。それらは[RandomLoRALoader](https://github.com/shin131002/RandomLoRALoader)を使ってください
- **LoRA Block Weight(LBW)は非対応です。** AnimaのフラットなTransformerブロック構造は、LBWが前提とするSD1.5/SDXL特有のIN/MID/OUTプリセットの考え方をそのまま適用できないため、現時点では対象外としています
- **`save_remapped`(ディスクへのリマップキャッシュ保存)は非搭載です。** これらのノードは実行のたびに異なるLoRAを選ぶ使い方が前提のため、キャッシュファイルは`manifest`/`extend_to_new_layers`/`blend_ratio`/`extend_strength`のいずれかを変更した瞬間に気づかれずに古くなってしまいますし、フォルダスキャン時にメタデータの無い「もう1つのLoRA候補」として再検出されてしまう問題もあります。そのため毎回その場でメモリ上のみでリマップします。LoRAファイル自体は小さいため、生成処理全体から見ればこの追加コストは無視できる範囲です
- **フォルダスキャン時に、ノード1〜4が書き出す`_animaremap<N>`・`_animaremap<N>_ext`形式のキャッシュファイルを除外します。** ノード1〜4を`save_remapped`ONで同じLoRAフォルダに対して使っていた場合、それらのキャッシュコピーは余分な候補として拾われず、自動的にスキップされます
- **リマップ関連設定はフォルダ/グループ単位ではなく共通1セットです。** `auto_remap`・`manifest`・`extend_to_new_layers`・`blend_ratio`・`extend_strength`は、ノード全体で共通の設定が使われます。ただし**実際に適用されるmanifestはLoRAごとに解決されます**。28層LoRAと40層LoRAが混在するフォルダを52層モデルに対して使う場合でも、`manifest`が`Auto`のままなら、それぞれ正しいmanifestで個別にリマップされます
- **層数不一致時に停止する挙動はノード1〜4と共通です。** ランダムに選ばれたLoRAが接続モデルより多い層を参照していた場合、部分的な適用にはせず、エラーを出して停止します

### 入力(Random LoRA Loader。Filteredは3フォルダの代わりに1フォルダ+キーワードフィルタになるだけで考え方は同じ)

| 名前 | 型 | 説明 |
|---|---|---|
| `model` / `clip` | MODEL / CLIP | ベースのモデル/CLIP |
| `additional_prompt_positive` / `_negative` | STRING | 追加プロンプト。選択された各LoRAのトリガーワードと結合される |
| `lora_folder_path_1..3` | STRING | グループごとのフォルダパス |
| `include_subfolders_1..3` | BOOLEAN | サブフォルダを含めるか |
| `unique_by_filename_1..3` | BOOLEAN | サブフォルダ間の同名ファイルを除外 |
| `model_strength_1..3` / `clip_strength_1..3` | STRING | 固定値(`"1.0"`)またはランダム範囲(`"0.4-0.8"`) |
| `num_loras_1..3` | INT | グループごとの選択数 |
| `trigger_word_source` | ドロップダウン | `json_combined`/`json_random`/`json_sample_prompt`/`metadata` |
| `seed` | INT | ランダム選択のシード値 |
| `auto_remap` | BOOLEAN | ノード全体で共通(デフォルトON) |
| `extend_to_new_layers` | BOOLEAN | 【実験的機能】ノード全体で共通(デフォルトOFF) |
| `blend_ratio` | FLOAT | ノード全体で共通(デフォルト1.0) |
| `extend_strength` | FLOAT | ノード全体で共通(デフォルト0.5) |
| `manifest` | ドロップダウン | デフォルトの`Auto (Recommended)`は、選択された各LoRAごとにmanifestを自動解決します。特定のファイルを選択すると、全LoRAでそれを強制します |

### 出力

元のRandomLoRALoaderノードと同じ構成です: `MODEL`、`CLIP`、`positive_text`、`negative_text`、`positive`(CONDITIONING)、`negative`(CONDITIONING)、`preview`(IMAGE)。`positive_text`には選択された各LoRAが`<lora:名前:model強度:clip強度>, トリガーワード`の形式で含まれます(CONDITIONING出力に渡す前にLoRA構文は除去されます)。

---

## Anima-3.8B(52層)対応について

[lylogummy/Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B)(52層)は、Anima-2.9Bをコミュニティが52ブロック(既存40ブロック+新規学習12ブロック)まで拡張したモデルで、プロンプト理解力向上のためのQwen3.5 4B cross-attentionコンポーネント(オプション)と対になっています。このコンポーネントはリリースによって、別ファイルのアダプターとして配布される場合(初期のpreview版)と、DiT本体のチェックポイントに同梱される場合(Anima-3.8B v1.1の"Semantic Connector v2"以降)があり、この配布形態は既に一度変わっており、今後さらに変わる可能性もあります。本パッケージが扱うのは52層チェックポイントの**DiTブロック構造のみ**です。どちらの形で配布されていても、Qwen3.5コンポーネントは独立した追加キー群(`semantic_attentions.*`など、どの`net.blocks.N`パターンにも一致しない関連キー)であり、リマップ処理は一切関与しません——検出・リマップともに`net.blocks.N`というキーのみを見ています。従来40層Anima-2.9BのDiT向けに作られたLoRA・マージ用モデルは、これまでのAnima-2.9B向けリマップと全く同じ考え方で52層のDiTにリマップされます。実機確認済み：Qwen3.5を完全に未接続にしても(ネイティブのQwen3 0.6B経路のみ)、生成は問題なく行えます。

> **出典について:** 公式のAnima→Anima-2.9B用`expand_manifest.json`とは異なり、Anima-2.9B→Anima-3.8Bの拡張については公式の対応表が公開されていません。`mapping/expand_manifest_40_52.json`は、`anima38B_base.safetensors`とAnima-2.9Bチェックポイントをブロック単位で比較(`self_attn.q_proj.weight`と`mlp.layer1.weight`のコサイン類似度、相互にクロスチェック済み)することで復元したものです。これはLLaMA Pro方式に典型的な「隣接ブロックをコピーしてから追加学習する」という初期化パターンを検出する手法です。今後公式の対応表が公開された場合はこのファイルを差し替え、`mapping/expand_manifest_28_52_composed.json`も再生成してください(下記参照)。

### ⚠️ テキストエンコーダーに関する留意点(Qwen3.5 4B)

本パッケージのリマップ処理が扱うのは**DiTのブロック構造のみ**です。接続するテキストエンコーダー経路が生成結果にどう影響するかには一切関与しません。これはブロックリマップとは完全に独立した軸の話ですが、Anima-3.8B特有の事情でここに明記しておく価値があると判断しました。

- **「Qwen3.5 4B adapterがそもそも機能するか」はブロック数(世代)の話であり、全世代で同じように機能します。** adapterが注入する先は、Animaの標準的なブロックテンプレートが元から持っている`cross_attn`スロットです。このスロットはAnima-3.8Bより前から存在し、ブロックリマップの有無によっても変わりません。52層に限った話ではなく、28層・40層でも同じように機能します。
- **「結果の良し悪し」は、ブロック数とは無関係な、テキストエンコーダー自体の特性の話です。** Qwen3.5 4Bは、Anima/Anima-2.9B用LoRAが本来学習時に前提としていたQwen3 0.6Bと比べて、格段に能力の高い言語モデルで、指示追従性もずっと強いです。あるLoRAやプロンプトがテキストエンコーダーを切り替えた際にどこまで通用するかを左右するのは、この能力差であって、「どの世代のDiTに繋いでいるか」ではありません。

これはLoRAがそもそもブロックリマップを必要としていたかどうかとは無関係です。ここで触れているのは、Anima-3.8Bがこのモデル系列で初めて「テキストエンコーダーを追加で選べる」世代になったことで、この区別が初めて意味を持つようになったからです。

## `expand_manifest.json`と複数世代対応について

`mapping/`内の各ファイルは、1つのAnima拡張ステップにおけるブロックインデックスの対応情報を記録しています(`old_block_count`、`new_block_count`、`insertion_positions`、そして各新規層が初期化時にどの旧層からコピーされたかを示す`inserted_to_source`)。各ノードの`manifest`ドロップダウンはデフォルトで`Auto (Recommended)`になっており、実際に扱っているLoRA/モデルのペアの`(old_block_count, new_block_count)`に一致するmanifestを自動選択するため、通常この項目を意識する必要はありません。

`mapping/expand_manifest_28_52_composed.json`は手作業で作ったものではなく、`mapping/scripts/compose_manifests.py`によって28→40と40→52のmanifestから機械的に合成されています。これにより、28層LoRAを52層モデルに1段階で直接リマップでき、ノード側に複雑な連鎖処理を持たせる必要がありません。

将来、新しいAnima世代が独自のブロック拡張とともに登場した場合:

1. その世代のmanifest(公式のものがあればそれを、無ければ同様の比較手法で復元したものを)`expand_manifest_<旧>_<新>.json`として`mapping/`に追加する
2. `python mapping/scripts/compose_manifests.py auto mapping/`を実行し、既存のmanifestとの新たな組み合わせを全て再生成する
3. それ以外は何も変更不要です。`Auto`により、全ノードが新しいmanifestを自動的に使うようになります

## 既知の制約

- 公式に公開されているブロック対応表は、従来Anima→Anima-2.9Bの拡張についてのみです。それ以降(現時点では→Anima-3.8B/52層)は復元データに依存します。詳細は上記の出典についての注記を参照してください
- LoRAのキー命名規則が想定外のパターンの場合、自動検出に失敗しリマップされません
- **世代の自動判定は、LoRAのキーが実際に参照している最大ブロック番号に基づいており、ファイルに埋め込まれたラベルを見ているわけではありません。** そのため、後の世代用に学習されたLoRAでも、前半〜中盤のブロックしか触っていない(例: 構造専用LoRA)場合は、誤って前の世代用と判定されリマップされてしまう可能性があります。現時点では特定の`manifest`ファイルを手動で強制する以外の上書き手段は用意していません。実運用で問題になるようであれば、「特定世代を強制」といった専用オプションの追加を検討します
- モデルマージは`comfy.model_patcher.ModelPatcher`の`get_key_patches`/`add_patches`(ComfyUI標準のマージ機構と同じ仕組み)を利用しています

## ライセンスについて

Anima(base)・Anima-2.9B・Anima-3.8B(52層)は、いずれも**CircleStone Labs Non-Commercial License**の下で提供されています(Anima-3.8Bのモデルカード自体に、元のAnima-base/Anima-2.9Bのライセンスに従う旨が明記されています)。本パッケージ(LoRA Remap / Model Mergeノード)を使って生成される、リマップ後のLoRAファイルやマージ後のモデルファイルは、対象がどの世代であっても、いずれもこのライセンスにおける「Derivative(派生物)」に該当するため、**同じ非商用制限が引き継がれます**。

- モデル本体・その派生物(今回のリマップ済みLoRA、マージ済みモデルを含む)は、非商用目的でのみ使用可能です
- 一方で、これらのモデルを使って**生成した画像(Outputs)自体は商用利用が可能**です(ライセンス上、生成画像は「Derivative」の定義から明示的に除外されています)
- Animaはさらに`Cosmos-Predict2-2B-Text2Image`の派生モデルにも該当するため、その範囲でNVIDIA Open Model License Agreementの条件も付随します
- ライセンス上、個人が「重みファイル(モデルやLoRA)そのもの」を有償配布すること自体は例外的に認められていますが(第2.c項)、それを組み込んだ製品・サービス・ツールとしての提供は対象外で、別途通常の商用ライセンスが必要です
- Anima-3.8Bに付随する別コンポーネントのQwen3.5 4Bテキストエンコーダー自体はApache 2.0です(このサイズ帯のQwenモデルの一般的なライセンスと同様)。本パッケージはこのファイルを一切処理・再配布しないため制約には関与しませんが、参考情報として付記しておきます

本ツールは個人利用・非商用での使用を前提としています。商用利用や配布を検討する場合は、必ず一次情報である`LICENSE.md`(Hugging Face上のAnimaリポジトリに同梱)や各モデルページのライセンス記載を確認するか、専門家に相談してください。本READMEの記載は法的助言ではありません。

なお、このライセンス制約は**Animaのモデル重み自体(および、それを使って作られたリマップ済みLoRA・マージ済みモデル)に適用されるもの**であり、本リポジトリの**コード自体(ノードのPython実装)はMITライセンス**(同梱の`LICENSE`ファイル参照)の下で公開しています。

## 免責事項とサポートポリシー

### 免責事項

- このノードは**技術サポートなし**で提供されます
- 機能の保証はありません
- 将来のComfyUIアップデートとの互換性は保証されません
- バグレポートや機能リクエストに対応しない場合があります
- 自己責任で使用してください

### サポート状況

- ❌ issueやメールでの個別サポートなし
- ❌ バグ修正や機能追加の保証なし
- ✅ コードはオープンソース - 自由にフォーク・修正可能
- ✅ コミュニティディスカッション歓迎(返答の約束なし)

### 問題の報告

サポートは保証されませんが、以下が可能です:
1. リポジトリの既存issueを確認
2. このREADMEとトラブルシューティングセクションを確認
3. issueを開く(対応されない場合があります)
4. 自分でフォークして修正

## ライセンス

MIT License - 自由に使用、変更、配布できます。

ただし前述の通り、Animaモデル自体(重み、およびそこから作られるLoRA・マージ済みモデル)は別途CircleStone Labs Non-Commercial Licenseの制約を受けます。
