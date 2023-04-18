const fs = require('fs');
const path = require('path');

function main() {
    const dataPath = process.argv[2];
    const trainPath = process.argv[3];
    const testPath = process.argv[4];

    fs.mkdirSync(testPath);
    fs.mkdirSync(trainPath);

    const countries = fs.readdirSync(dataPath);
    countries.forEach(country => {
        const slug = country.trim().toLowerCase().replace(/\s/g, '_');

        fs.readdirSync(path.join(dataPath, country)).forEach((imgName, i) => {
            const imgPath = path.join(dataPath, country, imgName);
            const extension = path.extname(imgName)
            const newPath = path.join(Math.random() > 0.8 ? testPath : trainPath,
                `${slug}_${i.toString().padStart(5, '0')}${extension}`);
            fs.renameSync(imgPath, newPath);
        });
    });
}

main();
