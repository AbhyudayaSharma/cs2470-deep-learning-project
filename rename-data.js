import {renameSync, readdirSync, mkdirSync} from 'fs';
import 'path';
import * as path from "path";

function main() {
    const dataPath = process.argv[2];
    const trainPath = process.argv[3];
    const testPath = process.argv[4];

    mkdirSync(testPath);
    mkdirSync(trainPath);

    const countries = readdirSync(dataPath);
    countries.forEach(country => {
        const slug = country.trim().toLowerCase().replaceAll(/\s/, '_');

        readdirSync(path.join(dataPath, country)).forEach((imgName, i) => {
            const imgPath = path.join(dataPath, country, imgName);
            const extension = path.extname(imgName)
            const newPath = path.join(Math.random() > 0.8 ? testPath : trainPath,
                `${slug}_${i.toString().padStart(3, '0')}.${extension}`);
            renameSync(imgPath, newPath);
        });
    });
}

main();
