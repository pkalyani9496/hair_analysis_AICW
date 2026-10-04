COLOR_PALETTE = {
    "Soft Black": {"hex": "#171717", "description": "Soft, natural-looking black"},
    "Jet Black": {"hex": "#101010", "description": "Deep, high-shine black"},
    "Blue Black": {"hex": "#1D2638", "description": "Black with a cool blue cast"},
    "Natural Black": {"hex": "#211B18", "description": "Balanced natural black"},
    "Mocha Brown": {"hex": "#4C2C2A", "description": "Cool, rich brunette"},
    "Espresso": {"hex": "#3B241C", "description": "Deep coffee brown"},
    "Chocolate Brown": {"hex": "#5A3825", "description": "Classic dimensional brunette"},
    "Dark Cherry": {"hex": "#541A2B", "description": "Subtle red-violet brunette"},
    "Burgundy": {"hex": "#6D1F3A", "description": "Deep wine red"},
    "Plum": {"hex": "#673147", "description": "Muted plum brunette"},
    "Mahogany": {"hex": "#6E2C2C", "description": "Red-brown depth"},
    "Dark Brown": {"hex": "#493329", "description": "Natural, dimensional brown"},
    "Chestnut": {"hex": "#7A4A2B", "description": "Warm chestnut brown"},
    "Auburn": {"hex": "#8A3324", "description": "Brown with copper-red warmth"},
    "Maple Brown": {"hex": "#8C5135", "description": "Soft maple warmth"},
    "Warm Brown": {"hex": "#7A4B34", "description": "Balanced warm brown"},
    "Toffee Brown": {"hex": "#9B684A", "description": "Soft golden brunette"},
    "Mushroom Brown": {"hex": "#8B7A69", "description": "Smoky neutral brunette"},
    "Smoky Brown": {"hex": "#6C6258", "description": "Cool, muted brown"},
    "Ash Brown": {"hex": "#74685B", "description": "Cool ash brunette"},
    "Caramel": {"hex": "#B97943", "description": "Warm caramel dimension"},
    "Cinnamon": {"hex": "#A55A36", "description": "Soft cinnamon copper"},
    "Copper": {"hex": "#B65E32", "description": "Classic copper warmth"},
    "Rust Copper": {"hex": "#B86135", "description": "Deep, earthy copper"},
    "Bronze": {"hex": "#A96F45", "description": "Muted bronze highlights"},
    "Rose Gold": {"hex": "#C9806E", "description": "Soft rosy warmth"},
    "Rosewood": {"hex": "#8C4A5B", "description": "Muted rose-brown"},
    "Red Violet": {"hex": "#8A3A5C", "description": "Cool berry red"},
    "Honey Blonde": {"hex": "#C99A52", "description": "Rich golden blonde"},
    "Golden Blonde": {"hex": "#D6B36A", "description": "Bright golden blonde"},
    "Sunlit Blonde": {"hex": "#D9B77A", "description": "Soft sunlit blonde"},
    "Sandy Blonde": {"hex": "#C4AE82", "description": "Natural sandy blonde"},
    "Beige Blonde": {"hex": "#C8B584", "description": "Balanced beige blonde"},
    "Champagne Blonde": {"hex": "#E6D0A6", "description": "Soft champagne blonde"},
    "Strawberry Blonde": {"hex": "#D88C6B", "description": "Golden blonde with a rose tint"},
    "Ash Blonde": {"hex": "#B7AA91", "description": "Cool, muted blonde"},
    "Silver Ash": {"hex": "#C5C8D0", "description": "Smoky silver ash"},
    "Platinum Blonde": {"hex": "#E7E1D6", "description": "Light, cool platinum"},
    "Cream Blonde": {"hex": "#F0E2C0", "description": "Warm, creamy blonde"},
    "Icy Blonde": {"hex": "#DDEAF1", "description": "Cool icy blonde"},
    "Purple": {"hex": "#6B2D8C", "description": "Vibrant royal purple"},
    "Deep Violet": {"hex": "#4A154B", "description": "Rich jewel-toned violet"},
    "Lavender": {"hex": "#9B7BB8", "description": "Soft smoky lavender pastel"},
    "Pastel Pink": {"hex": "#D97B93", "description": "Soft muted rose pastel"},
}

HAIRCUT_RECOMMENDATIONS = {
    "Curly Hair": [
        {"name": "Curly Shag", "description": "Crown layers add shape while keeping curl movement.", "image": "/haircuts/curly_shag.jpg"},
        {"name": "Rounded Layers", "description": "Soft layers help curls form an even silhouette.", "image": "/haircuts/rounded_layers.jpg"},
        {"name": "Curly Bob", "description": "A chin-to-shoulder shape that shows curl definition.", "image": "/haircuts/curly_bob.jpg"},
        {"name": "Tapered Cut", "description": "Keeps length on top with a lighter, shaped outline.", "image": "/haircuts/tapered_cut.jpg"},
    ],
    "Straight Hair": [
        {"name": "Blunt Lob", "description": "A clean edge makes straight hair look fuller.", "image": "/haircuts/blunt_lob.jpg"},
        {"name": "Long Face-Framing Layers", "description": "Adds movement without losing overall length.", "image": "/haircuts/long_face_framing_layers.jpg"},
        {"name": "French Bob", "description": "A short, precise shape with a strong outline.", "image": "/haircuts/french_bob.jpg"},
        {"name": "Curtain Fringe", "description": "A soft front shape that pairs with long or medium lengths.", "image": "/haircuts/curtain_fringe.jpg"},
    ],
    "Wavy Hair": [
        {"name": "Butterfly Layers", "description": "Long and short layers bring out natural wave movement.", "image": "/haircuts/butterfly_layers.jpg"},
        {"name": "Textured Lob", "description": "A shoulder-length cut that keeps waves light.", "image": "/haircuts/textured_lob.jpg"},
        {"name": "Long Shag", "description": "Layering adds definition and an intentionally undone finish.", "image": "/haircuts/long_shag.jpg"},
        {"name": "Curtain Fringe", "description": "Frames the face while blending into loose waves.", "image": "/haircuts/curtain_fringe.jpg"},
    ],
}

COLOR_RECOMMENDATIONS_BY_TONE = {
    "Black": ["Mocha Brown", "Burgundy", "Chestnut", "Copper"],
    "Dark Brown": ["Chestnut", "Mahogany", "Caramel", "Rose Gold"],
    "Medium Brown": ["Caramel", "Copper", "Honey Blonde", "Auburn"],
    "Light Brown": ["Caramel", "Copper", "Golden Blonde", "Rose Gold"],
    "Dark Blonde": ["Honey Blonde", "Ash Blonde", "Champagne Blonde", "Copper"],
    "Light / Blonde": ["Beige Blonde", "Platinum Blonde", "Rose Gold", "Strawberry Blonde"],
}

COLOR_RECOMMENDATIONS_BY_TYPE = {
    "Curly Hair": ["Burgundy", "Copper", "Caramel", "Rosewood"],
    "Straight Hair": ["Mushroom Brown", "Beige Blonde", "Toffee Brown", "Ash Blonde"],
    "Wavy Hair": ["Caramel", "Honey Blonde", "Maple Brown", "Rose Gold"],
}


def recommend_haircuts(hair_type):
    return list(HAIRCUT_RECOMMENDATIONS.get(hair_type, HAIRCUT_RECOMMENDATIONS["Wavy Hair"]))


def recommend_colors(hair_type, current_color, limit=5):
    tone_colors = COLOR_RECOMMENDATIONS_BY_TONE.get(
        current_color,
        COLOR_RECOMMENDATIONS_BY_TONE["Medium Brown"],
    )
    type_colors = COLOR_RECOMMENDATIONS_BY_TYPE.get(
        hair_type,
        COLOR_RECOMMENDATIONS_BY_TYPE["Wavy Hair"],
    )
    ordered_names = list(dict.fromkeys([*type_colors[:2], *tone_colors, *type_colors[2:]]))
    return [
        {"name": name, **COLOR_PALETTE[name]}
        for name in ordered_names[:limit]
    ]
