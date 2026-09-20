import React from 'react';

const MediaMismatch = () => {

    window.onbeforeunload = null;

    return(
        <div style={{
            height: "100vh",
            display: "flex",
            flexDirection: "column"}}>
            <div style={{margin: "auto"}}>
                <div className="section">
                    <div className="content" style={{maxWidth: "600px"}}>
                        <h5 style={{color: "red"}}>Important participation notice</h5>
                        <p>
                            Dear participant,
                        </p>
                        <p>
                            Thank you for your interest in this study. Unfortunately, the study cannot be displayed to you right now. Processing the study requires a minimum screen size and the study can only be processed with a computer mouse and keyboard, but not with touch. Participation on a smartphone or tablet is therefore not possible.
                        </p>
                        <p>
                            Please try to do the study on a larger screen and use a computer mouse and keyboard. Thank you for your understanding. Also, make sure that the zoom of your web browser is set to 100%.
                        </p>
                        <p>
                            If you have any questions about the study or if you meet the technical requirements mentioned above, but the study is still not siplayed to you, you can contactus at any time.
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

export default MediaMismatch;