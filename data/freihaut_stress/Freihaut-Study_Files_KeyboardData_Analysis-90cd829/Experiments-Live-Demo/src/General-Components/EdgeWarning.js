import React from 'react';

import Firefox_logo from './firefox.png';
import Chro_logo from './chrome.png';

const EdgeWarning = (props) => {

    window.onbeforeunload = null;

    return(
        <div style={{height: "100vh", display: "flex", flexDirection: "column", fontSize: "large"}}>
            <div style={{margin: "auto"}}>
                <div className="section">
                    <div className="content" style={{maxWidth: "900px"}}>
                        <h5 style={{color: "red"}}>Important participation notice</h5>
                        <p>
                            Dear participant,
                        </p>
                        <p>
                            Thank you for your interest in this study! Before you start the study, we would like to inform you that your internet browser with which you have just accessed this study may not support all of the study’s features.
                        </p>
                        <p>
                            If it is possible for you, we recommend that you access the study in one of the following two internet browsers:
                        </p>
                        <ul style={{listStyleType: "none"}}>
                            <li><span style={{marginRight: "10px"}} className="icon"><img src={Firefox_logo} alt={"Firefox-Logo"}/></span><strong>Mozilla Firefox</strong></li>
                            <li><span style={{marginRight: "10px"}} className="icon"><img src={Chro_logo} alt={"Chrome-Logo"}/></span><strong>Google Chrome</strong></li>
                        </ul>
                        <p>
                            If you are unable to access the study in another internet browser, you can still participate in the study now, but you should be prepared that some parts of the study may not be displayed optimally.
                        </p>
                        <p>
                            If you have any questions, please feel free to contact us at any time.
                        </p>
                        <p>
                            Paul Freihaut, M.Sc. <br/>
                            University of Freiburg <br/>
                            Department of Psychology<br/>
                            Occupational and Consumer Psychology<br/>
                            Engelbergerstraße 41 <br/>
                            D-79085 Freiburg <br/>
                            E-Mail: <a href={"mailto:paul.freihaut@psychologie.uni-freiburg.de"}>paul.freihaut@psychologie.uni-freiburg.de</a>
                        </p>
                        <div style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            marginTop: "58px",
                        }} className="control">
                            <button className="button is-link" onClick={() => props.startExpWithEdge()}
                            >Participate in study</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )

}

export default EdgeWarning;