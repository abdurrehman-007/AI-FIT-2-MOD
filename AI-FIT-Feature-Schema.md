# AI-FIT feature-wise database schema

The local launcher defaults to SQLite and local authentication. SQLite stores application records in `documents(collection TEXT, id TEXT, body TEXT NOT NULL)`, with primary key `(collection, id)`. The fields below live inside the JSON `body`. With Firestore enabled, each collection name becomes a Firestore collection and each `id` becomes a document key. Authentication is configured independently.

| Feature | Collection / document key | Fields and types |
|---|---|---|
| Accounts | `users/{uid}` | `id`, `email`, `name` (string), `store` (store ID or null). Local accounts also have `password` (hash string) and optional `password_version` (string). Firebase Auth manages its own credentials. |
| Stores and settings | `stores/{storeId}` | `id`, `name`, `chart_status` (string), `owner` (user ID or null for the demo), `chart` (legacy cm map), `charts` (category map in inches), `baseline_orders`, `baseline_returns` (integer). |
| Catalog and inventory | `products/{productId}` | `id`, `store`, `sku`, `name`, `category`, `color`, `image` (string), `price` (number in PKR), `stock` (integer), `sizes` (string array), `sample`, `active` (boolean). `image` is an image ID or empty string. |
| Uploads and generated previews | `images/{imageId}` | `id`, `kind` (person, garment, or result), `owner` (user ID), `created` (Unix seconds). Image bytes live in the server's `instance/images` directory by default, including when metadata uses Firestore. |
| Try-on, sizing and retail analytics | `events/{eventId}` | `id`, `kind`, `store` (string), `time` (Unix seconds). Fields vary by event type, as shown below. |
| Local sign-in throttling | `limits/{uid}` | `id` (string), `attempts` (integer), `until` (Unix seconds). |
| Local reset-email throttling | `reset_limits/{uid}` | `minute` (integer, Unix minutes). |
| QR and kiosk | Uses the existing store and account records | QR payload is generated from the store ID and configured public URL. No separate QR or kiosk collection. |

## Event fields

| `kind` | Additional fields |
|---|---|
| `tryon` | `product` (product ID), `category` (string), `seconds` (number), `result` (image ID), optional `customer_key` (pseudonymous string). |
| `tryon_failed` | `product` (product ID), `seconds` (number). |
| `size` | `product` (product ID), `category`, `size`, `segment` (string). Raw shopper measurements are not stored. |
| `sale`, `return` | `reference` (order reference), `tryon` (try-on event ID or empty string). |

## Size chart fields

Path: `stores/{storeId}.charts.{category}.{size}`. Sizes are XS, S, M, L, XL and XXL. All chart values are numeric body measurements in inches.

| Category | Required fields | Optional fields |
|---|---|---|
| `upper` | `chest`, `waist` | `shoulder`, `sleeve` |
| `lower` | `waist`, `hips` | `inseam`, `thigh` |
| `overall` | `chest`, `waist`, `hips` | `shoulder`, `sleeve`, `inseam`, `thigh` |

References: `users.store → stores.id`, `stores.owner → users.id`, `products.store → stores.id`, `products.image → images.id`, `images.owner → users.id`, `events.store → stores.id`, `events.product → products.id`, `events.result → images.id`, and `events.tryon → events.id`.
