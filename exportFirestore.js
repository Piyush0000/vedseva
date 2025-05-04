const fs = require("fs");
const admin = require("firebase-admin");

// Path to your service account key
const serviceAccount = require("./serviceAccountKey.json");

// Initialize Firebase Admin
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const db = admin.firestore();

async function exportData() {
  const collections = await db.listCollections();
  const data = {};

  for (const col of collections) {
    const snapshot = await col.get();
    data[col.id] = {};
    snapshot.forEach((doc) => {
      data[col.id][doc.id] = doc.data();
    });
  }

  fs.writeFileSync("firestore-export.json", JSON.stringify(data, null, 2));
  console.log("Exported to firestore-export.json");
}

exportData();
