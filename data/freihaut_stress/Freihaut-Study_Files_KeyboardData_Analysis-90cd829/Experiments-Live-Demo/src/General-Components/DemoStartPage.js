import React from 'react';

import Firefox_logo from './firefox.png';
import Chro_logo from './chrome.png';
import IE_logo from './IE-Icon.png';
import Edge_logo from './Edge-Icon.png';

const expBoxStyle = {
    display: "block",
    padding: "1rem",
    boxShadow: "0 2px 3px rgba(10,10,10,.1),0 0 0 1px rgba(10,10,10,.1)"
}


const DemoStartPage = (props) => {

    window.onbeforeunload = null;

    return(
        <div>
            <div style={{maxWidth: "1000px", margin: "0px auto", fontSize: "large"}}>
                <div className="section">
                    <div className="content">
                        <h4>
                            Welcome!
                        </h4>
                        <p>
                            On this page, you can try out the experimental apps presented in the research paper "Does People’s Keyboard Typing Reflect their Stress Level – An Explorative Study"
                            by Freihaut & Göritz. The source code of this app is available on <a href="https://github.com/Freihaut/Experiments-Live-Demo" target="_blank" rel="noreferrer">GitHub</a> and is
                            additionally archived in a data repository.
                        </p>
                        <p>
                            The experimental apps on this page are demo versions. They are identical to the
                            original study apps in their composition and functionality, but they do not save any data.
                            All data collected during the experiments can be viewed in the browser console. In the original
                            study apps, we used  <a href="https://firebase.google.com/" target="_blank" rel="noreferrer">Firebase</a> to to save the data
                            instead of logging the data in the console.
                        </p>
                        <p>
                            On top of this page, you find a bar with buttons to navigate through the experimental apps.
                            You can skip to the next experimental page, go back to the previous experimental page or
                            exit an experiment and go back to this introduction page. In the original study apps, there
                            was no navigation bar.
                        </p>
                        <p>
                            The original apps were in German. For the demos, we translated all texts into English. You can
                            find a transcript with the original German study texts for each experiment with their respective English
                            translation (as used in the demos) in the study repository.
                        </p>
                        <p>
                            <u>
                                Technical Information:
                            </u>
                        </p>
                        <p>
                            The experimental apps were not designed to work on a smartphone and tablet. They have a fixed
                            size and are not responsive. If your screen size is too small, you will be warned and you won't
                            be able to use the apps. If you do not have a larger screen to view the experiments, you can adjust the zoom level in
                            your web browser.
                        </p>
                        <p>
                            The experimental apps were tested to work in recent versions of <span style={{marginRight: "5px"}}><img style={{width: "18px", height: "18px"}} src={Firefox_logo} alt={"Firefox-Logo"}/></span>
                            <strong>Mozilla Firefox</strong> and <span style={{marginRight: "5px"}}><img style={{width: "18px", height: "18px"}} src={Chro_logo} alt={"Chrome-Logo"}/></span><strong>Google Chrome</strong>.
                            In other web browsers, the apps might not work as intended. It is not possible to do the experiments
                            in the <span style={{marginRight: "5px"}}><img style={{width: "18px", height: "18px"}} src={IE_logo} alt={"IE-Logo"}/></span><strong>Internet Explorer</strong>.
                            If you try to start the experiments in an older version of <span style={{marginRight: "5px"}}><img style={{width: "18px", height: "18px"}} src={Edge_logo} alt={"Edge-Logo"}/></span><strong>Microsoft Edge</strong>,
                            you will receive a warning.
                        </p>
                        <p>
                           For questions and comments, please contact:
                        </p>
                        <p>
                            Paul Freihaut<br/>
                            University of Freiburg <br/>
                            Department of Occupational and Consumer Psychology<br/>
                            E-Mail: <a href={"mailto:paul.freihaut@psychologie.uni-freiburg.de"}>paul.freihaut@psychologie.uni-freiburg.de</a>
                        </p>
                        <section>
                            <div className="container">
                                <div className="columns">
                                    <div className="column">
                                        <div style={expBoxStyle}>
                                            <p className="title is-4">Online Study</p>
                                            <p className="title is-5">Screensize requirements:</p>
                                            <p className="subtitle is-6">Width: 1024px<br/>Height: 600px</p>
                                            <div style={{
                                                marginTop: "25px",
                                            }} className="control">
                                                <button style={{width: "100px"}} className="button is-link" onClick={() => props.startExperiment("OnlineStudy")}
                                                >Start</button>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="column">
                                        <div style={expBoxStyle}>
                                            <p className="title is-4">Labratory Study</p>
                                            <p className="title is-5">Screensize requirements:</p>
                                            <p className="subtitle is-6">Width: 1150px<br/>Height: 700px</p>
                                            <div style={{
                                                marginTop: "25px",
                                            }} className="control">
                                                <button style={{width: "100px"}} className="button is-link" onClick={() => props.startExperiment("LabStudy")}
                                                >Start</button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </section>
                    </div>
                </div>
            </div>
        </div>
    )

}

export default DemoStartPage;