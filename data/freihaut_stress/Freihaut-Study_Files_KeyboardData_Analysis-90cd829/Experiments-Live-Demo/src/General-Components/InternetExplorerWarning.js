import React from 'react';

import Firefox_logo from './firefox.png';
import Chro_logo from './chrome.png';

const InternetExplorerWarning = () => {

    window.onbeforeunload = null;

    return(
        <div>
            <div style={{maxWidth: "900px", margin: "0px auto", fontSize: "large"}}>
                <div className="section">
                    <div className="content">
                        <h5 style={{color: "red"}}>Important participation notice</h5>
                        <p>
                            Dear participant,
                        </p>
                        <p>
                            Thank you for your interest in this study! Unfortunately, the study cannot be displayed to you. Your program (Internet Explorer) with which you have opened this study does not support all the functions that are necessary for the study to be carried out properly.
                        </p>
                        <p>
                            If you which to participate in the study, we recommend that you access the study with one of the following two internet browsers:
                        </p>
                        <ul style={{listStyleType: "none"}}>
                            <li><span style={{marginRight: "10px"}}><img style={{width: "32px", height: "32px"}} src={Firefox_logo} alt={"Firefox-Logo"}/></span><strong>Mozilla Firefox</strong></li>
                            <li><span style={{marginRight: "10px"}}><img style={{width: "32px", height: "32px"}} src={Chro_logo} alt={"Chrome-Logo"}/></span><strong>Google Chrome</strong></li>
                        </ul>
                        <p>
                            Thank you for your understanding. If you have any questions, please feel free to contact us at any time.
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
                    </div>
                </div>
            </div>
        </div>
    )

}

export default InternetExplorerWarning;